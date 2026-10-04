# Paired oversampling study

This directory contains the reproducible paired Monte Carlo experiment used
to study how the oversampling factor affects the existing SVD-ESPRIT Prony
pipeline.

## Main design

- model order: `n = 4`
- oversampling factors: `rho = 1, ..., 10`
- noise levels: 10 equally spaced values from 0 to 0.5
- trials: 1000 per noise level
- noise model: real Gaussian noise
- pairing: trial `i` uses `RandomState(i)`; within a trial, every `rho`
  uses a progressively longer prefix of the same noise realization
- estimator: the existing `prony.prony_method`
- component matching: the existing Hungarian pole matching in
  `prony.match_estimates`
- common reconstruction scoring window: first `2n+1 = 9` samples

The paired design is intentionally separate from
`experiments/run_experiments.py`, which uses independent random streams
across configurations.

## Run

From the repository root:

```bash
python -m experiments.paired_oversampling.run
```

Outputs are written to:

```text
results/paired_oversampling/
    raw_results.npz
    summary.csv
```

`raw_results.npz` stores trial-level results. `summary.csv` is generated
from those raw values and contains the mean, standard deviation, median,
quartiles, and 95th percentile for each metric.

## Current metrics

- relative exponent error
- relative amplitude error
- noisy-target RMSE on the common nine-sample window
- clean-target RMSE on the common nine-sample window
- condition number of the full noisy Hankel matrix

Additional diagnostics should be added as separate, explicit analyses rather
than changing this baseline experiment.
