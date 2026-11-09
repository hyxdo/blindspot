"""Does fine-tuning on a model's mistakes beat fine-tuning on random samples?

The experiment perturbs a labeled training pool, finds what the pretrained model
gets wrong, then trains two versions with the same number of examples: one on
misclassified examples, one on a random sample of the pool. Both are scored on
held-out images, clean and perturbed, because robustness gained at the cost of
clean accuracy is a trade rather than a win.

Holding out a perturbation family keeps it out of training entirely, which tests
whether the gains carry over to conditions the model never saw.

Only the last ResNet block and head are trained, so everything before them gives
the same output for every model. Those outputs are computed once and reused,
which is what makes the comparison practical without a GPU.
"""

from __future__ import annotations

import copy
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, models

from .model import to_tensor
from .perturb import FAMILIES, Perturbation, apply, sweep_grid

# ImageNet indices of the ten Imagenette classes, in torchvision's label order.
# Scoring only these logits means the pretrained head can be fine-tuned as is.
IMAGENETTE_INDICES = (0, 217, 482, 491, 497, 566, 569, 571, 574, 701)
DEFAULT_UNFROZEN = ("layer4", "fc")
ARCHES = ("resnet18", "resnet34", "resnet50")
BOOTSTRAP_SAMPLES = 2000

Strategy = Literal["hard", "random"]
STRATEGIES: tuple[Strategy, ...] = ("hard", "random")
Log = Callable[[str], None]


@dataclass(frozen=True)
class Example:
    image: int
    perturbation: Perturbation


class PerturbedDataset(Dataset):
    def __init__(
        self,
        images: Sequence[np.ndarray],
        labels: Sequence[int],
        examples: Sequence[Example],
        preprocess: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.images = images
        self.labels = labels
        self.examples = list(examples)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        example = self.examples[index]
        image = apply(self.images[example.image], example.perturbation)
        return self.preprocess(to_tensor(image)), int(self.labels[example.image])


def split_resnet(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Split a ResNet into the frozen part and the part that gets trained."""
    prefix = nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3
    )
    suffix = nn.Sequential(
        OrderedDict(
            layer4=model.layer4, avgpool=model.avgpool, flatten=nn.Flatten(1), fc=model.fc
        )
    )
    return prefix.eval(), suffix.eval()


def join_resnet(base: nn.Module, suffix: nn.Module) -> nn.Module:
    """Return a full copy of `base` that uses a fine-tuned suffix."""
    model = copy.deepcopy(base)
    model.layer4 = suffix.layer4
    model.fc = suffix.fc
    return model


@torch.no_grad()  # not inference_mode: these outputs later feed a training graph
def extract_features(
    prefix: nn.Module,
    dataset: Dataset,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> TensorDataset:
    features, targets = [], []
    for batch, labels in DataLoader(dataset, batch_size=batch_size):
        features.append(prefix(batch.to(device)).cpu())
        targets.append(labels)
    return TensorDataset(torch.cat(features), torch.cat(targets))


def build_pool(
    num_images: int, grid: Sequence[Perturbation], per_image: int, rng: np.random.Generator
) -> list[Example]:
    """Pair each image with `per_image` distinct perturbations drawn from the grid."""
    count = min(per_image, len(grid))
    return [
        Example(image, grid[int(choice)])
        for image in range(num_images)
        for choice in rng.choice(len(grid), size=count, replace=False)
    ]


def training_grid(grid: Sequence[Perturbation], holdout: str | None) -> list[Perturbation]:
    """Return the perturbations allowed in training."""
    if holdout is None:
        return list(grid)
    if holdout not in FAMILIES:
        raise ValueError(f"unknown perturbation family {holdout!r}")
    return [p for p in grid if p.family != holdout]


def select_indices(
    wrong: np.ndarray, budget: int, strategy: Strategy, rng: np.random.Generator
) -> np.ndarray:
    """Pick pool positions for one strategy, drawing without replacement."""
    if strategy == "hard":
        candidates = np.flatnonzero(wrong)
    elif strategy == "random":
        candidates = np.arange(len(wrong))
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    if budget > len(candidates):
        raise ValueError(f"budget {budget} exceeds the {len(candidates)} candidates available")
    return rng.choice(candidates, size=budget, replace=False)


@torch.inference_mode()
def predict(
    model: nn.Module,
    dataset: Dataset,
    class_indices: Sequence[int],
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted class, label) arrays, scoring only `class_indices`."""
    model.eval()
    subset = torch.as_tensor(list(class_indices), device=device)
    predictions, targets = [], []
    for batch, labels in DataLoader(dataset, batch_size=batch_size):
        logits = model(batch.to(device))[:, subset]
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def finetune(
    base: nn.Module,
    dataset: Dataset,
    class_indices: Sequence[int],
    epochs: int = 3,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 0,
    unfrozen: Sequence[str] = DEFAULT_UNFROZEN,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Return a fine-tuned copy of `base`; only `unfrozen` submodules are trained."""
    torch.manual_seed(seed)
    model = copy.deepcopy(base).to(device)
    params = []
    for name, param in model.named_parameters():
        param.requires_grad_(any(name == p or name.startswith(f"{p}.") for p in unfrozen))
        if param.requires_grad:
            params.append(param)
    if not params:
        raise ValueError(f"no parameters match {list(unfrozen)}")

    # Stay in eval mode: a few hundred examples shouldn't rewrite BatchNorm statistics.
    model.eval()
    subset = torch.as_tensor(list(class_indices), device=device)
    # SGD on purpose: Adam's first steps move every weight by about `lr` regardless
    # of gradient size, which wrecked pretrained accuracy in a single update.
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    for _ in range(epochs):
        for batch, labels in loader:
            optimizer.zero_grad()
            logits = model(batch.to(device))[:, subset]
            nn.functional.cross_entropy(logits, labels.to(device)).backward()
            optimizer.step()
    return model


def scores_by_image(correct: np.ndarray, grid: Sequence[Perturbation]) -> dict[str, np.ndarray]:
    """Turn a conditions x images correctness table into per-image scores.

    Row 0 is the clean image; the remaining rows follow `grid`.
    """
    scores = {"clean": correct[0].astype(float), "perturbed": correct[1:].mean(axis=0)}
    for family in FAMILIES:
        rows = [row for row, p in enumerate(grid, start=1) if p.family == family]
        if rows:
            scores[family] = correct[rows].mean(axis=0)
    return scores


def bootstrap_difference(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, samples: int = BOOTSTRAP_SAMPLES
) -> dict[str, float]:
    """Return the mean of a - b with a 95% interval, resampling held-out images."""
    difference = a - b
    picks = rng.integers(0, len(difference), size=(samples, len(difference)))
    means = difference[picks].mean(axis=1)
    return {
        "difference": float(difference.mean()),
        "low": float(np.percentile(means, 2.5)),
        "high": float(np.percentile(means, 97.5)),
    }


def compare_scores(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray], rng: np.random.Generator
) -> dict[str, dict[str, float]]:
    return {metric: bootstrap_difference(a[metric], b[metric], rng) for metric in a}


def summarize(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": float(np.mean([run[metric] for run in runs])),
            "std": float(np.std([run[metric] for run in runs])),
        }
        for metric in runs[0]
    }


@dataclass(frozen=True)
class ExperimentConfig:
    out: str = "results/experiment"
    data_root: str = "data"
    arch: str = "resnet18"
    pool_images: int = 2000
    per_image: int = 3
    eval_images: int = 300
    budget: int = 500
    epochs: int = 3
    lr: float = 1e-3
    batch_size: int = 32
    seeds: int = 3
    holdout_family: str | None = None
    save_checkpoints: bool = False


def compare_strategies(
    base: nn.Module,
    preprocess: Callable[[torch.Tensor], torch.Tensor],
    pool_images: Sequence[np.ndarray],
    pool_labels: Sequence[int],
    eval_images: Sequence[np.ndarray],
    eval_labels: Sequence[int],
    class_indices: Sequence[int],
    config: ExperimentConfig,
    grid: Sequence[Perturbation] | None = None,
    log: Log = print,
    device: str | torch.device = "cpu",
    on_model: Callable[[str, int, nn.Module], None] | None = None,
) -> dict:
    """Train both strategies and score everything on the held-out images."""
    grid = sweep_grid() if grid is None else list(grid)
    allowed = training_grid(grid, config.holdout_family)
    rng = np.random.default_rng(1)
    prefix, suffix = split_resnet(base)
    pool = build_pool(len(pool_images), allowed, config.per_image, rng)

    log(f"computing frozen features for {len(pool)} pool examples")
    pool_features = extract_features(
        prefix, PerturbedDataset(pool_images, pool_labels, pool, preprocess), device=device
    )
    predicted, targets = predict(suffix, pool_features, class_indices, device=device)
    wrong = predicted != targets
    train_examples = min(config.budget, int(wrong.sum()))
    if train_examples == 0:
        raise RuntimeError("the baseline made no mistakes on the pool, so there is nothing to compare")
    if train_examples < config.budget:
        log(f"only {train_examples} misclassified examples; lowering both budgets to match")

    conditions = [Perturbation(), *grid]
    evaluation = [Example(i, p) for p in conditions for i in range(len(eval_images))]
    log(f"computing frozen features for {len(evaluation)} held-out examples")
    eval_features = extract_features(
        prefix, PerturbedDataset(eval_images, eval_labels, evaluation, preprocess), device=device
    )

    def score(model: nn.Module) -> dict[str, np.ndarray]:
        guesses, truth = predict(model, eval_features, class_indices, device=device)
        return scores_by_image((guesses == truth).reshape(len(conditions), len(eval_images)), grid)

    baseline = score(suffix)
    mean_scores: dict[str, dict[str, np.ndarray]] = {}
    strategies: dict[str, dict] = {}
    for strategy in STRATEGIES:
        runs = []
        for seed in range(config.seeds):
            log(f"fine-tuning {strategy}, seed {seed + 1} of {config.seeds}")
            chosen = select_indices(wrong, train_examples, strategy, np.random.default_rng(seed))
            tuned = finetune(
                suffix,
                Subset(pool_features, chosen.tolist()),
                class_indices,
                epochs=config.epochs,
                lr=config.lr,
                batch_size=config.batch_size,
                seed=seed,
                device=device,
            )
            runs.append(score(tuned))
            if on_model is not None:
                on_model(strategy, seed, join_resnet(base, tuned))
        means = [{metric: float(values.mean()) for metric, values in run.items()} for run in runs]
        strategies[strategy] = {"runs": means, "summary": summarize(means)}
        mean_scores[strategy] = {
            metric: np.mean([run[metric] for run in runs], axis=0) for metric in runs[0]
        }

    interval_rng = np.random.default_rng(2)
    return {
        "config": asdict(config),
        "pool_size": len(pool),
        "pool_wrong": int(wrong.sum()),
        "train_examples": train_examples,
        "eval_images": len(eval_images),
        "conditions": len(grid),
        "baseline": {metric: float(values.mean()) for metric, values in baseline.items()},
        "strategies": strategies,
        "comparisons": {
            "hard - random": compare_scores(mean_scores["hard"], mean_scores["random"], interval_rng),
            "hard - baseline": compare_scores(mean_scores["hard"], baseline, interval_rng),
            "random - baseline": compare_scores(mean_scores["random"], baseline, interval_rng),
        },
    }


def render_report(results: dict) -> str:
    holdout = results["config"].get("holdout_family")
    metrics = ["clean", *[f for f in FAMILIES if f in results["baseline"]], "perturbed"]
    titles = {"clean": "clean", "perturbed": "all perturbed"}

    def title(metric: str) -> str:
        return f"{metric} (never trained on)" if metric == holdout else titles.get(metric, metric)

    header = "| model | " + " | ".join(title(m) for m in metrics) + " |"
    divider = "|---|" + "---|" * len(metrics)
    seeds = results["config"]["seeds"]
    lines = [
        "# Blindspot experiment",
        "",
        f"Each model trained on {results['train_examples']} examples, drawn from a pool of "
        f"{results['pool_size']} perturbed images ({results['pool_wrong']} misclassified by the baseline). "
        f"Scored on {results['eval_images']} held-out images under {results['conditions']} perturbations. "
        f"{seeds} seed{'s' if seeds != 1 else ''} per strategy.",
        "",
        "## Accuracy (mean ± std across seeds)",
        "",
        header,
        divider,
        "| baseline | " + " | ".join(f"{results['baseline'][m]:.3f}" for m in metrics) + " |",
    ]
    for strategy in STRATEGIES:
        summary = results["strategies"][strategy]["summary"]
        cells = [f"{summary[m]['mean']:.3f} ± {summary[m]['std']:.3f}" for m in metrics]
        lines.append(f"| {strategy} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Differences (95% bootstrap interval over held-out images)",
        "",
        "An interval that excludes zero is evidence of a real difference on images like these.",
        "",
        header.replace("| model |", "| comparison |"),
        divider,
    ]
    for name, comparison in results["comparisons"].items():
        cells = [
            f"{comparison[m]['difference']:+.3f} [{comparison[m]['low']:+.3f}, {comparison[m]['high']:+.3f}]"
            for m in metrics
        ]
        lines.append(f"| {name.replace(' - ', ' − ')} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def load_imagenette(
    root: str, split: str, limit: int, rng: np.random.Generator
) -> tuple[list[np.ndarray], list[int]]:
    """Load a random subset of Imagenette into memory, downloading it if needed."""
    Path(root).mkdir(parents=True, exist_ok=True)
    download = not (Path(root) / "imagenette2-160").exists()
    dataset = datasets.Imagenette(root, split=split, size="160px", download=download)
    chosen = rng.choice(len(dataset), size=min(limit, len(dataset)), replace=False)
    images, labels = [], []
    for index in chosen:
        image, label = dataset[int(index)]
        images.append(np.asarray(image.convert("RGB")).copy())
        labels.append(int(label))
    return images, labels


def run_experiment(config: ExperimentConfig, log: Log = print, device: str | None = None) -> dict:
    if config.arch not in ARCHES:
        raise ValueError(f"the experiment supports {', '.join(ARCHES)}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)

    weights = models.get_model_weights(config.arch).DEFAULT
    base = models.get_model(config.arch, weights=weights).eval().to(device)
    rng = np.random.default_rng(0)
    log("loading Imagenette")
    pool_images, pool_labels = load_imagenette(config.data_root, "train", config.pool_images, rng)
    eval_images, eval_labels = load_imagenette(config.data_root, "val", config.eval_images, rng)

    def save(strategy: str, seed: int, model: nn.Module) -> None:
        (out / "checkpoints").mkdir(exist_ok=True)
        torch.save(model.state_dict(), out / "checkpoints" / f"{strategy}_seed{seed}.pt")

    results = compare_strategies(
        base,
        weights.transforms(),
        pool_images,
        pool_labels,
        eval_images,
        eval_labels,
        IMAGENETTE_INDICES,
        config,
        log=log,
        device=device,
        on_model=save if config.save_checkpoints else None,
    )
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "report.md").write_text(render_report(results), encoding="utf-8")
    log(f"wrote {out / 'report.md'}")
    return results
