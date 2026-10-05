# Paired oversampling study

This directory contains the reproducible CMMAI paired Monte Carlo study for
the SVD-ESPRIT Prony pipeline.

## Canonical main design

- model order: `n = 4`
- oversampling factors: `rho = 1, ..., 10`
- noise levels: 10 equally spaced values from 0 to 0.5
- trials: 1000 per noise level
- primary noise model: real Gaussian noise
- canonical RNG: trial `i` uses `numpy.random.default_rng(i)`
- pairing across `rho`: progressively longer prefixes of the same
  standardized noise realization are reused
- the same standardized realization is also reused across noise levels and
  scaled by the requested `sigma`
- estimator: the existing `prony.prony_method`
- common reconstruction scoring window: first `2n+1 = 9` samples

For oversampling factor `rho`, the study uses a
`(rho*n + 1) x (n + 1)` Hankel matrix and
`N = n*(rho + 1) + 1` samples.

The experiment deliberately couples sample count, observation horizon, and
Hankel aspect ratio. Results should therefore be interpreted as properties of
this complete oversampling design rather than as a universal statement about
one isolated mechanism.

## Primary parameter metrics

The final CMMAI analysis uses the audit runner for parameter-error reporting.

The primary exponent metric is wrap-aware. Imaginary exponent differences are
wrapped to `[-pi, pi)` before the relative complex-exponent error is formed.
Component matching is also performed with the corresponding wrapped-exponent
cost.

The primary amplitude metric is the relative amplitude error after this
wrapped-exponent matching.

The historical direct exponent error is retained in the audit outputs only for
provenance and reconciliation with the accepted abstract.

## Canonical experiment sequence

From the repository root, run:

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

For smoke tests, the Monte Carlo runners accept `--trials 3`.

Generated outputs live under:

```text
results/paired_oversampling/
```

The `results/` directory is git-ignored. The version-controlled source code
is sufficient to regenerate the numerical outputs.

## Role of each runner

### `run.py`

Produces the paper-level reconstruction and full noisy-Hankel diagnostics:

- noisy-target RMSE on the common nine-sample window
- clean-target RMSE on the common nine-sample window
- condition number of the full noisy Hankel matrix

It also stores historical parameter errors for continuity, but the final paper
uses the wrap-aware audit metrics for parameter-error claims.

### `audit_metrics.py`

Runs the full paired grid under either real or circular complex Gaussian noise
and stores:

- historical exponent error
- wrap-aware exponent error
- historical amplitude error
- wrapped-match amplitude error
- branch-jump count
- historical/wrapped matching disagreement

The circular-noise convention is

`eps = sigma/sqrt(2) * (xi + i*eta)`,

so `E|eps|^2 = sigma^2`.

### `diagnostics.py`

Stores three supporting diagnostics under the canonical real-noise protocol:

- damping RMSE
- wrapped frequency RMSE in cycles/sample
- oracle-pole amplitude error

The oracle-pole calculation fixes the true exponents and reruns only the
column-scaled Vandermonde least-squares amplitude stage. It is therefore a
diagnostic of how much ordinary amplitude deterioration is associated with
pole-estimation error. It is not a claim that Vandermonde conditioning is the
sole or dominant bottleneck.

### `spectral_diagnostics.py`

Uses cross-`rho` comparable spectral diagnostics:

- maximum and mean right-singular-subspace principal angle
- clean `sigma_n`
- noisy `sigma_n/sigma_(n+1)` separation ratio
- Hankel perturbation norm
- perturbation norm divided by clean `sigma_n`

The right singular subspace lives in the fixed ambient dimension `n+1`.
Cross-`rho` left-subspace angles are intentionally excluded because their
ambient dimension changes with `rho`.

These quantities support interpretation only. They are not a causal proof and
do not define an automatic rule for choosing `rho`.

### `scope_checks.py`

Runs separate `n=6` and `n=8` purely oscillatory scope checks. Because
these models have purely imaginary exponents, the reported pole metric is
wrapped frequency RMSE in cycles/sample rather than the main damped study's
relative complex-exponent error.

These are scope checks, not evidence for a universal model-order law.

### `signal_decay.py`

Provides a deterministic view of how the damped signal and individual
component envelopes change as the observation horizon grows. This is useful
for interpreting why later added samples may contain less signal information.

### `make_tables.py` and `make_figures.py`

These consume only the canonical outputs listed above. Before regeneration,
they clear old paper-facing CSV or PNG files so stale outputs cannot remain
mixed with current ones.

Primary parameter plots/tables use the wrap-aware audit metrics. Reconstruction
and condition-number outputs come from `run.py`. Spectral plots/tables come
from `spectral_diagnostics.py`.

## Interpretation guardrails

- `rho=1` is simply the smallest tested member of the same pipeline, not a
  universal baseline method.
- Exact integer grid minima are descriptive and should not be presented as
  uniquely optimal values.
- A decreasing full noisy-Hankel condition number does not by itself imply
  better ESPRIT parameter recovery. ESPRIT does not directly invert the full
  Hankel matrix, and noise can lift the smallest singular value.
- The small noisy-target RMSE at `rho=1` partly reflects that the common
  nine-sample scoring window is also the complete `rho=1` fitting window.
- Right-subspace and spectral diagnostics are supporting evidence, not causal
  proof.
- The useful oversampling range depends on signal structure, noise level, and
  the quantity being estimated.

## Repository hygiene

This study requires Python 3.10 or newer, consistent with the repository
package metadata. Generated arrays, tables, and figures remain local under
`results/` and are excluded from version control.
