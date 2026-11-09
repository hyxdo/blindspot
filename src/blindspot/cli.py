"""Command line entry point: sweep, cases, replay, experiment, app."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Sequence

from .cases import DEFAULT_ROOT


def _sweep(args: argparse.Namespace) -> int:
    from .cases import CaseStore, read_image
    from .model import Classifier
    from .sweep import sweep

    classifier = Classifier.pretrained(args.arch, args.checkpoint)
    image = read_image(args.image)
    result = sweep(image, classifier, true_label=args.label)
    print(f"clean: {result.clean[0].label} ({result.clean[0].prob:.1%})")
    for row in result.rows:
        marker = "FAIL" if row.failed else "ok  "
        print(f"{marker}  {row.perturbation.describe():<20} {row.top[0].label} ({row.top[0].prob:.1%})")
    print(f"{len(result.failures)} of {len(result.rows)} settings failed")
    if args.save:
        store = CaseStore(args.store)
        for row in result.failures:
            case = store.save(image, row.perturbation, classifier, true_label=args.label)
            print(f"saved {case.id}")
    return 0


def _cases(args: argparse.Namespace) -> int:
    from .cases import CaseStore

    for case in CaseStore(args.store).all():
        print(
            f"{case.id}  {case.model['arch']:<20} {case.settings.describe():<24} "
            f"{case.clean_top[0]['label']} -> {case.perturbed_top[0]['label']}"
        )
    return 0


def _replay(args: argparse.Namespace) -> int:
    from .cases import CaseStore
    from .model import Classifier

    store = CaseStore(args.store)
    cases = [store.load(case_id) for case_id in args.ids] if args.ids else store.all()
    if not cases:
        print(f"no saved cases in {store.root}")
        return 0

    classifiers: dict[str, Classifier] = {}
    all_reproduced = True
    for case in cases:
        key = json.dumps(case.model, sort_keys=True)
        if key not in classifiers:
            classifiers[key] = Classifier.pretrained(case.model["arch"], case.model["checkpoint"])
        result = store.replay(case, classifiers[key])
        all_reproduced &= result.reproduced
        status = "reproduced" if result.reproduced else "CHANGED"
        print(
            f"{case.id}  {status:<10}  stored={result.stored_top1}  now={result.replayed_top1}  "
            f"max prob delta={result.max_prob_delta:.4f}"
        )
        if not result.model_matches:
            print("    model weights differ from the ones that produced this case")
        for name, (old, new) in result.environment_changes.items():
            print(f"    {name}: {old} -> {new}")
    return 0 if all_reproduced else 1


def _experiment(args: argparse.Namespace) -> int:
    from .finetune import ExperimentConfig, render_report, run_experiment

    names = {field.name for field in fields(ExperimentConfig)}
    results = run_experiment(ExperimentConfig(**{k: v for k, v in vars(args).items() if k in names}))
    print(render_report(results))
    return 0


def _app(args: argparse.Namespace) -> int:
    env = {**os.environ, "BLINDSPOT_CASES": str(args.store)}
    script = Path(__file__).with_name("app.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(script)], env=env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blindspot", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler, help_text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text)
        sub.set_defaults(handler=handler)
        return sub

    sweep = add("sweep", _sweep, "run one image through the perturbation grid")
    sweep.add_argument("image")
    sweep.add_argument("--arch", default="resnet18")
    sweep.add_argument("--checkpoint", help="fine-tuned weights to load")
    sweep.add_argument("--label", help="true ImageNet class name, if known")
    sweep.add_argument("--save", action="store_true", help="save every failure as a case")
    sweep.add_argument("--store", default=DEFAULT_ROOT)

    cases = add("cases", _cases, "list saved failures")
    cases.add_argument("--store", default=DEFAULT_ROOT)

    replay = add("replay", _replay, "regenerate saved failures; exits 1 if any changed")
    replay.add_argument("ids", nargs="*", help="case ids; all cases if omitted")
    replay.add_argument("--store", default=DEFAULT_ROOT)

    experiment = add("experiment", _experiment, "compare hard-example and random fine-tuning")
    experiment.add_argument("--out", default="results/experiment")
    experiment.add_argument("--data-root", default="data")
    experiment.add_argument("--arch", default="resnet18")
    experiment.add_argument("--pool-images", type=int, default=2000)
    experiment.add_argument("--per-image", type=int, default=3)
    experiment.add_argument("--eval-images", type=int, default=1000)
    experiment.add_argument("--budget", type=int, default=500)
    experiment.add_argument("--epochs", type=int, default=3)
    experiment.add_argument("--lr", type=float, default=1e-3)
    experiment.add_argument("--batch-size", type=int, default=32)
    experiment.add_argument("--seeds", type=int, default=3)
    experiment.add_argument("--save-checkpoints", action="store_true")

    app = add("app", _app, "open the interactive tool")
    app.add_argument("--store", default=DEFAULT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)
