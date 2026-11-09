# Blindspot experiment

Each model trained on 130 examples, drawn from a pool of 1600 perturbed images (130 misclassified by the baseline). Scored on 150 held-out images under 34 perturbations. 3 seeds per strategy.

## Accuracy (mean ± std across seeds)

| model | clean | lighting | blur (never trained on) | framing | all perturbed |
|---|---|---|---|---|---|
| baseline | 0.987 | 0.922 | 0.588 | 0.908 | 0.857 |
| hard | 0.904 ± 0.028 | 0.808 ± 0.036 | 0.481 ± 0.009 | 0.820 ± 0.028 | 0.755 ± 0.027 |
| random | 0.969 ± 0.006 | 0.906 ± 0.013 | 0.614 ± 0.031 | 0.907 ± 0.010 | 0.855 ± 0.013 |

## Differences (95% bootstrap interval over held-out images)

An interval that excludes zero is evidence of a real difference on images like these.

| comparison | clean | lighting | blur (never trained on) | framing | all perturbed |
|---|---|---|---|---|---|
| hard − random | -0.064 [-0.096, -0.036] | -0.098 [-0.132, -0.067] | -0.133 [-0.175, -0.092] | -0.086 [-0.119, -0.056] | -0.099 [-0.131, -0.068] |
| hard − baseline | -0.082 [-0.124, -0.044] | -0.114 [-0.152, -0.079] | -0.107 [-0.165, -0.049] | -0.088 [-0.127, -0.051] | -0.101 [-0.140, -0.065] |
| random − baseline | -0.018 [-0.040, +0.004] | -0.016 [-0.028, -0.004] | +0.027 [-0.004, +0.055] | -0.002 [-0.015, +0.011] | -0.002 [-0.014, +0.010] |
