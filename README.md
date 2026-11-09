# Blindspot

Image classifiers can be right about a photo and wrong about the same photo taken a little darker, a little blurrier, or framed a little differently. Blindspot helps you find those conditions, save them so they can be reproduced exactly, and test whether fine-tuning fixes them.

It has two parts:

- **An interactive tool** for changing lighting, blur, and framing with sliders while watching a pretrained model's predictions, then saving what breaks it.
- **An experiment** that fine-tunes a pretrained classifier on its own mistakes or on random samples, then compares robustness and clean accuracy on held-out images.

Built with Python, PyTorch, NumPy, OpenCV, and Streamlit.

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The first run downloads the pretrained weights from torchvision.

## Interactive tool

```bash
blindspot app
```

Add an image, move the sliders, and watch the original and perturbed predictions side by side. Set the true class if you know it; otherwise Blindspot flags any change from the clean prediction. **Run sweep** tries a standard grid of settings, and **Save this case** records a failure.

## Finding and replaying failures from the command line

```bash
blindspot sweep photo.jpg --label "golf ball" --save
blindspot cases
blindspot replay
```

`replay` regenerates each saved image and runs the recorded model again. It exits with status 1 if any prediction moved, so it can also serve as a regression check.

### What a saved case contains

| | |
|---|---|
| Source image | Lossless PNG, verified against a hash of its pixels |
| Settings | Every perturbation value |
| Model | Architecture, weights, and a checksum of any fine-tuned checkpoint |
| Predictions | Top-5 classes and confidences, clean and perturbed |
| Environment | Python, NumPy, OpenCV, PyTorch, and torchvision versions |

Perturbations use no randomness, so the settings and source image fully determine the perturbed pixels. Blur and shift are measured as fractions of the image size, so a setting means the same thing at any resolution.

## Fine-tuning experiment

```bash
blindspot experiment --out results/experiment
```

The run uses [Imagenette](https://github.com/fastai/imagenette), a ten-class subset of ImageNet, downloaded on first use (about 100 MB).

The default settings score roughly 250,000 images, so use a GPU. On a CPU, shrink the run with `--pool-images`, `--eval-images`, `--budget`, and `--seeds`, and treat results from very small runs as a check that the pipeline works, not as findings.

1. Apply random perturbations from the sweep grid to images from the training split.
2. Find which of those examples the pretrained model gets wrong.
3. Train two kinds of model on the **same number of examples**, repeated across seeds: `hard` draws from the misclassified examples, `random` draws from the whole pool.
4. Score the baseline and both kinds of model on the validation split, clean and under every perturbation in the grid.

Only the last ResNet block and the classifier head are trained, and BatchNorm statistics stay frozen. Results are written to `results.json` and `report.md`.

Things to keep in mind when reading the results:

- Imagenette comes from ImageNet, so the pretrained model may have seen some of these images. Absolute accuracy can run high; the comparison between models is the useful part.
- The held-out images are new, but the held-out perturbations come from the same grid used for training. This measures recovery from known conditions, not robustness to unseen ones.

## Tests

```bash
pytest
```

The tests use a tiny stand-in model, so they run quickly and need no downloads.

## Roadmap

- Hold out whole perturbation families from training to measure transfer to unseen conditions
- Sweep combinations of settings instead of one at a time
- Load custom labeled datasets
