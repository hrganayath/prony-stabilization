# Paired oversampling study

This directory contains the source code for the paired Monte Carlo study
**“Noise-Dependent Oversampling Trade-Offs in ESPRIT-Based Prony Recovery of
Exponential Sums.”** The curated output snapshot is published separately under
[`reproducibility/paired_oversampling/`](../../reproducibility/paired_oversampling/README.md).

## Canonical main design

- model order: `n = 4`
- oversampling factors: `rho = 1, ..., 10`
- noise levels: 10 equally spaced values from 0 to 0.5
- trials: 1000 per noise level
- primary noise model: real Gaussian noise
- audit noise model: circular complex Gaussian noise
- canonical RNG: trial `i` uses `numpy.random.default_rng(i)`
- pairing across `rho`: progressively longer prefixes of the same
  standardized noise realization are reused
- the same standardized realization is reused across noise levels and scaled
  by the requested `sigma`
- estimator: `prony.prony_method`
- common reconstruction scoring window: first `2n+1 = 9` samples

For oversampling factor `rho`, the study uses a
`(rho*n + 1) x (n + 1)` Hankel matrix and
`N = n*(rho + 1) + 1` samples. Thus sample count, observation horizon, and
Hankel aspect ratio change together. Interpret the results as properties of
this complete design, not as an isolated causal effect of matrix shape.

## Installation and tests

From the repository root, install the package and development dependencies:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Canonical experiment sequence

Run from the repository root:

```bash
python -m experiments.paired_oversampling.run
python -m experiments.paired_oversampling.audit_metrics --noise-model real
python -m experiments.paired_oversampling.audit_metrics --noise-model circular
python -m experiments.paired_oversampling.summarize_audit
python -m experiments.paired_oversampling.diagnostics
python -m experiments.paired_oversampling.spectral_diagnostics
python -m experiments.paired_oversampling.summarize_spectral
python -m experiments.paired_oversampling.scope_checks
python -m experiments.paired_oversampling.signal_decay
python -m experiments.paired_oversampling.make_tables
python -m experiments.paired_oversampling.make_figures
```

The repository also contains supporting analyses for paired uncertainty,
fixed-sample-count shape controls, and damping/frequency sensitivity. These
are separate diagnostic analyses rather than steps needed to regenerate the
primary canonical grid:

```bash
python -m experiments.paired_oversampling.bootstrap_analysis
python -m experiments.paired_oversampling.uncertainty_analysis
python -m experiments.paired_oversampling.controls_uncertainty
python -m experiments.paired_oversampling.fixed_n_shape_control
python -m experiments.paired_oversampling.analyze_fixed_n_shape
python -m experiments.paired_oversampling.damping_sweep
python -m experiments.paired_oversampling.damping_sweep_closefreq
python -m experiments.paired_oversampling.damping_sweep_closefreq_uncertainty
```

The Monte Carlo runners that expose `--trials` accept a reduced value for a
smoke run. A smoke run is not a substitute for the canonical 1000-trial data.

## Output locations and published snapshot

Working outputs are generated under:

```text
results/paired_oversampling/
```

The `results/` directory is git-ignored to prevent accidental publication of
intermediate or stale outputs. The reviewed snapshot is tracked separately at:

```text
reproducibility/paired_oversampling/
  data/                 NPZ trial-level arrays and metadata
  tables/summaries/     summary, audit, diagnostic, and uncertainty CSVs
  tables/formatted/     formatted CSV tables
  figures/              PNG figures
```

The snapshot README describes each file family and the array metadata. To
refresh the public snapshot after intentionally regenerating and reviewing
outputs, copy only the final files into the corresponding `reproducibility/`
subdirectories. Do not publish the entire `results/` working directory.

## Metric definitions and protocol

The final parameter metric is wrap-aware: imaginary exponent differences are
wrapped to `[-pi, pi)` before the relative complex-exponent error is formed.
Component matching uses the corresponding wrapped-exponent cost. The primary
amplitude metric is relative amplitude error after wrapped-exponent matching.
Historical direct exponent and amplitude errors are retained in audit outputs
for provenance and reconciliation, not as the final primary metrics.

For circular complex Gaussian noise the convention is
`eps = sigma/sqrt(2) * (xi + i*eta)`, so `E|eps|^2 = sigma^2`.

## Role of the main runners

- `run.py`: common-window noisy-target and clean-target reconstruction RMSE,
  plus full noisy-Hankel condition diagnostics.
- `audit_metrics.py`: real/circular noise parameter audits, wrap-aware errors,
  historical metrics, branch-jump counts, and matching disagreement.
- `summarize_audit.py`: selected summaries and grid-minimum summaries from the
  audit datasets.
- `diagnostics.py`: damping RMSE, wrapped frequency RMSE, and oracle-pole
  amplitude error. The oracle-pole calculation isolates the amplitude-fit
  stage diagnostically; it does not establish a sole causal bottleneck.
- `spectral_diagnostics.py` and `summarize_spectral.py`: comparable right
  singular-subspace and singular-value/perturbation diagnostics. The right
  subspace has fixed ambient dimension `n+1`; changing-dimension left-subspace
  angles are not used for cross-`rho` comparisons.
- `scope_checks.py`: separate `n=6` and `n=8` purely oscillatory scope checks.
- `signal_decay.py`: deterministic signal and component-envelope decay over
  the observation horizon.
- `make_tables.py` and `make_figures.py`: paper-facing tables and figures from
  the canonical outputs. They clear stale generated CSV/PNG outputs before
  rebuilding them.

The additional analysis runners listed above create bootstrap/uncertainty,
fixed-`N`, and damping-sweep outputs. Consult their docstrings and source for
the precise comparison definitions and design choices.

## Interpretation guardrails

- `rho=1` is the smallest tested member of the same pipeline, not a universal
  baseline method.
- Exact integer grid minima are descriptive; bootstrap comparisons show why
  favourable ranges are more defensible than claiming a unique optimum.
- A decreasing full noisy-Hankel condition number does not by itself imply
  better ESPRIT parameter recovery. ESPRIT does not directly invert the full
  Hankel matrix, and noise can lift its smallest singular value.
- The small noisy-target RMSE at `rho=1` partly reflects that the common
  nine-sample scoring window is also the complete `rho=1` fitting window.
- Fixed-`N` controls are a separate diagnostic. Their design and metrics should
  not be conflated with the primary common-nine-sample scoring protocol.
- Spectral and oracle-pole diagnostics support interpretation; they are not
  causal proofs or automatic rules for selecting `rho`.
- The useful oversampling range depends on signal structure, noise level, and
  the target quantity (exponent, amplitude, or reconstruction).
