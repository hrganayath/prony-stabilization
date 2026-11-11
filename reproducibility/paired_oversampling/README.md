# Curated numerical outputs

This directory contains the reviewed numerical snapshot associated with the
paired oversampling study. It is version-controlled separately from the local
`results/` working directory so intermediate and stale generated files are
not accidentally published.

## Contents

| Directory | Files | Purpose |
|---|---:|---|
| `data/` | 14 NPZ | Trial-level numeric arrays and embedded design metadata |
| `tables/summaries/` | 32 CSV | Aggregate summaries, audits, bootstrap/uncertainty results, diagnostics, and sweep summaries |
| `tables/formatted/` | 18 CSV | Pivoted/formatted tables for inspection and reporting |
| `figures/` | 18 PNG | Main results, diagnostic, control, and signal-decay figures |

The counts refer to the curated snapshot at publication time. The files are
provided for direct inspection and downstream analysis; installing the
package does not trigger Monte Carlo computation.

## NPZ datasets

All arrays were checked to load with `numpy.load(..., allow_pickle=False)` and
contained finite numeric values in the inspected snapshot. The principal
arrays and shapes are:

| File | Main array(s) | Shape |
|---|---|---|
| `raw_results.npz` | `raw` | `(10, 10, 1000, 5)` |
| `audit_real.npz`, `audit_circular.npz` | `raw` | `(10, 10, 1000, 6)` each |
| `estimates_real.npz`, `estimates_circular.npz` | `amplitudes`, `exponents` | `(10, 10, 1000, 4)` each |
| `diagnostics_raw.npz` | `raw` | `(10, 10, 1000, 3)` |
| `spectral_diagnostics.npz` | `raw` | `(10, 10, 1000, 6)` |
| `damping_sweep.npz`, `damping_sweep_closefreq.npz` | `raw` | `(4, 2, 10, 1000, 4)` each |
| `fixed_n_shape_N17.npz` | `raw` | `(2, 3, 1000, 5)` |
| `fixed_n_shape_N29.npz` | `raw` | `(2, 6, 1000, 5)` |
| `fixed_n_shape_N45.npz` | `raw` | `(2, 10, 1000, 5)` |
| `scope_n6.npz` | `raw` | `(2, 10, 1000, 2)` |
| `scope_n8.npz` | `raw` | `(2, 10, 1000, 2)` |

The NPZ files also store the relevant coordinate vectors and metadata, such as
noise levels, oversampling factors, trial seeds, metric names, noise model,
model order, or signal parameters. Inspect `z.files` and the accompanying
metadata arrays for the exact names in each dataset; array dimension order is
not inferred from a CSV table.

## CSV organization

- `tables/summaries/`: long-form per-grid summaries, selected-noise summaries,
  grid minima, paired bootstrap comparisons, tail summaries, uncertainty
  intervals, fixed-`N` comparisons, damping sweeps, scope checks, and signal
  decay data.
- `tables/formatted/`: wide/pivoted tables intended for convenient comparison
  and reporting. These are derived views; the long-form CSVs and NPZ files
  retain the supporting data.

Some related information appears in both long-form and formatted form. This is
intentional: the two representations serve different inspection and reporting
uses. They should not be treated as independent experimental replications.

## Figures

`figures/` contains all 18 PNG figures from the local finalized snapshot,
including parameter-error curves, reconstruction trade-offs, spectral
subspace diagnostics, noise-model comparison, fixed-`N` controls, and signal
and component decay plots. Figure filenames identify the plotted quantity and
selected noise level where applicable.

## Reproduction and provenance

The design, deterministic trial-seed convention, paired-prefix noise protocol,
metric definitions, runner commands, and interpretation limitations are
specified in [`../../experiments/paired_oversampling/README.md`](../../experiments/paired_oversampling/README.md).
The source of truth for generating new outputs is the code under
`experiments/paired_oversampling/`; this snapshot is the curated set of outputs
from the archived local run. Newly generated working files go under the
Git-ignored `results/paired_oversampling/` directory and should be reviewed
before replacing this public snapshot.
