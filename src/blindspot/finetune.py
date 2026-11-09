"""Does fine-tuning on a model's mistakes beat fine-tuning on random samples?

The experiment perturbs a labeled training pool, finds what the pretrained model
gets wrong, then trains two versions with the same number of examples: one on
misclassified examples, one on a random sample of the pool. Both are scored on
held-out images, clean and perturbed, because robustness gained at the cost of
clean accuracy is a trade rather than a win.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models

from .model import to_tensor
from .perturb import FAMILIES, Perturbation, apply, sweep_grid

# ImageNet indices of the ten Imagenette classes, in torchvision's label order.
# Scoring only these logits means the pretrained head can be fine-tuned as is.
IMAGENETTE_INDICES = (0, 217, 482, 491, 497, 566, 569, 571, 574, 701)
DEFAULT_UNFROZEN = ("layer4", "fc")

Strategy = Literal["hard", "random"]
STRATEGIES: tuple[Strategy, ...] = ("hard", "random")


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


def select_examples(
    pool: Sequence[Example],
    wrong: np.ndarray,
    budget: int,
    strategy: Strategy,
    rng: np.random.Generator,
) -> list[Example]:
    """Pick the training set for one strategy, drawing without replacement."""
    if strategy == "hard":
        candidates = np.flatnonzero(wrong)
    elif strategy == "random":
        candidates = np.arange(len(pool))
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    if budget > len(candidates):
        raise ValueError(f"budget {budget} exceeds the {len(candidates)} candidates available")
    return [pool[int(i)] for i in rng.choice(candidates, size=budget, replace=False)]


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
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    for _ in range(epochs):
        for batch, labels in loader:
            optimizer.zero_grad()
            logits = model(batch.to(device))[:, subset]
            nn.functional.cross_entropy(logits, labels.to(device)).backward()
            optimizer.step()
    return model


def evaluate(
    model: nn.Module,
    images: Sequence[np.ndarray],
    labels: Sequence[int],
    preprocess: Callable[[torch.Tensor], torch.Tensor],
    class_indices: Sequence[int],
    grid: Sequence[Perturbation],
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Score clean accuracy and accuracy under every perturbation in the grid."""
    conditions = [Perturbation(), *grid]
    examples = [Example(i, p) for p in conditions for i in range(len(images))]
    dataset = PerturbedDataset(images, labels, examples, preprocess)
    predicted, targets = predict(model, dataset, class_indices, batch_size, device)
    correct = (predicted == targets).reshape(len(conditions), len(images))

    by_family: dict[str, list[float]] = defaultdict(list)
    for perturbation, row in zip(grid, correct[1:]):
        by_family[perturbation.family].append(float(row.mean()))
    return {
        "clean": float(correct[0].mean()),
        "perturbed": float(correct[1:].mean()),
        **{family: float(np.mean(by_family[family])) for family in FAMILIES if family in by_family},
    }


@dataclass(frozen=True)
class ExperimentConfig:
    out: str = "results/experiment"
    data_root: str = "data"
    arch: str = "resnet18"
    pool_images: int = 2000
    per_image: int = 3
    eval_images: int = 1000
    budget: int = 500
    epochs: int = 3
    lr: float = 1e-3
    batch_size: int = 32
    seeds: int = 3
    save_checkpoints: bool = False


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


def summarize(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": float(np.mean([run[metric] for run in runs])),
            "std": float(np.std([run[metric] for run in runs])),
        }
        for metric in runs[0]
    }


def render_report(results: dict) -> str:
    metrics = ["clean", "perturbed", *FAMILIES]
    lines = [
        "# Blindspot experiment",
        "",
        f"Trained on {results['train_examples']} examples per model "
        f"({results['pool_size']} in the pool, {results['pool_wrong']} misclassified by the baseline). "
        f"Scored on {results['eval_images']} held-out images, averaged over "
        f"{results['config']['seeds']} seeds (mean ± std).",
        "",
        "| model | " + " | ".join(metrics) + " |",
        "|---|" + "---|" * len(metrics),
        "| baseline | " + " | ".join(f"{results['baseline'][m]:.3f}" for m in metrics) + " |",
    ]
    for strategy in STRATEGIES:
        summary = results["strategies"][strategy]["summary"]
        cells = [f"{summary[m]['mean']:.3f} ± {summary[m]['std']:.3f}" for m in metrics]
        lines.append(f"| {strategy} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run_experiment(
    config: ExperimentConfig, log: Callable[[str], None] = print, device: str | None = None
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)

    weights = models.get_model_weights(config.arch).DEFAULT
    base = models.get_model(config.arch, weights=weights).eval().to(device)
    preprocess = weights.transforms()
    grid = sweep_grid()

    # One fixed pool, so seeds differ only in which examples are drawn and in SGD.
    rng = np.random.default_rng(0)
    log("loading Imagenette")
    pool_images, pool_labels = load_imagenette(config.data_root, "train", config.pool_images, rng)
    eval_images, eval_labels = load_imagenette(config.data_root, "val", config.eval_images, rng)

    pool = build_pool(len(pool_images), grid, config.per_image, rng)
    log(f"scoring the baseline on {len(pool)} perturbed pool examples")
    predicted, targets = predict(
        base, PerturbedDataset(pool_images, pool_labels, pool, preprocess),
        IMAGENETTE_INDICES, device=device,
    )
    wrong = predicted != targets
    train_examples = min(config.budget, int(wrong.sum()))
    if train_examples == 0:
        raise RuntimeError("the baseline made no mistakes on the pool, so there is nothing to compare")
    if train_examples < config.budget:
        log(f"only {train_examples} misclassified examples; lowering both budgets to match")

    log(f"evaluating the baseline on {len(eval_images)} held-out images")
    score = lambda model: evaluate(
        model, eval_images, eval_labels, preprocess, IMAGENETTE_INDICES, grid, device=device
    )
    results: dict = {
        "config": asdict(config),
        "pool_size": len(pool),
        "pool_wrong": int(wrong.sum()),
        "train_examples": train_examples,
        "eval_images": len(eval_images),
        "baseline": score(base),
        "strategies": {},
    }

    for strategy in STRATEGIES:
        runs = []
        for seed in range(config.seeds):
            log(f"fine-tuning {strategy}, seed {seed}")
            chosen = select_examples(
                pool, wrong, train_examples, strategy, np.random.default_rng(seed)
            )
            model = finetune(
                base,
                PerturbedDataset(pool_images, pool_labels, chosen, preprocess),
                IMAGENETTE_INDICES,
                epochs=config.epochs,
                lr=config.lr,
                batch_size=config.batch_size,
                seed=seed,
                device=device,
            )
            runs.append(score(model))
            if config.save_checkpoints:
                (out / "checkpoints").mkdir(exist_ok=True)
                torch.save(model.state_dict(), out / "checkpoints" / f"{strategy}_seed{seed}.pt")
        results["strategies"][strategy] = {"runs": runs, "summary": summarize(runs)}

    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "report.md").write_text(render_report(results), encoding="utf-8")
    log(f"wrote {out / 'report.md'}")
    return results
