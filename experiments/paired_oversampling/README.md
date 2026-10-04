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

## Baseline run

From the repository root:

```bash
python -m experiments.paired_oversampling.run
```

For a quick smoke test:

```bash
python -m experiments.paired_oversampling.run --trials 3
```

Full-run outputs are written to:

```text
results/paired_oversampling/
    raw_results.npz
    summary.csv
```

The baseline raw array stores trial-level values for:

- relative exponent error
- relative amplitude error
- noisy-target RMSE on the common nine-sample window
- clean-target RMSE on the common nine-sample window
- condition number of the full noisy Hankel matrix

## Mechanism diagnostics

The baseline experiment is left unchanged. Additional diagnostics are run
separately with the same signal, noise levels, seeds, and paired-prefix
protocol:

```bash
python -m experiments.paired_oversampling.diagnostics
```

For a quick smoke test:

```bash
python -m experiments.paired_oversampling.diagnostics --trials 3
```

The diagnostic study records:

- damping RMSE from the real parts of the matched exponents
- wrapped frequency RMSE in cycles per sample
- oracle-pole amplitude error, obtained by fixing the true exponents and
  solving only the column-scaled Vandermonde least-squares problem
- largest principal angle between the clean and noisy rank-`n` left
  singular subspaces
- mean principal angle between those subspaces

Angles are reported in degrees. These quantities are supporting diagnostics;
they do not replace the baseline parameter and reconstruction errors.

Full diagnostic outputs are:

```text
results/paired_oversampling/
    diagnostics_raw.npz
    diagnostics_summary.csv
```

All summary CSV files are generated from the corresponding raw trial-level
arrays and contain the mean, standard deviation, median, quartiles, and 95th
percentile.


## Signal-decay diagnostic

A deterministic companion analysis shows how the damped signal and the
individual component envelopes evolve as the observation horizon grows:

```bash
python -m experiments.paired_oversampling.signal_decay
```

This analysis does not claim a formal SNR for the complex-valued observation
model.  It reports the clean-signal magnitude, component envelopes, and simple
magnitude-to-noise-standard-deviation ratios at representative noise levels.
It also marks the last sample included by each oversampling factor.

Outputs:

```text
results/paired_oversampling/
    signal_decay.csv
    signal_decay.png
```
