# Prony Stabilization: numerical reproducibility repository

[![Tests](https://github.com/hrganayath/prony-stabilization/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/hrganayath/prony-stabilization/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

This repository contains the Python implementation, experiment runners, tests,
and curated numerical outputs for the study **“Noise-Dependent Oversampling
Trade-Offs in ESPRIT-Based Prony Recovery of Exponential Sums.”** The study
examines how oversampling affects recovery in a four-term damped exponential
model under controlled paired-noise Monte Carlo experiments.

The repository is also the software companion to the author's work on
numerical stabilization of Prony-type exponential parameter estimation.

## What is included

- `src/prony/`: the reusable Prony/ESPRIT implementation and supporting
  utilities.
- `experiments/paired_oversampling/`: canonical experiment, audit, diagnostic,
  uncertainty, table, and figure-generation scripts.
- `reproducibility/paired_oversampling/`: the curated numerical snapshot
  (trial-level arrays, CSV summaries/tables, and PNG figures) used for review
  and inspection without rerunning the Monte Carlo calculations.
- `tests/`: automated tests for the package and research protocols.
- `.github/workflows/`: continuous integration for the test suite.

The manuscript and paper-preparation files are maintained separately and are
not part of this public repository.

## Installation

Python 3.10 or newer is required. From a clone of this repository:

```bash
git clone https://github.com/hrganayath/prony-stabilization.git
cd prony-stabilization
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run the tests

```bash
pytest -q
```

The package uses a `src/` layout. Installing it in editable mode as above
ensures `prony` is importable by the tests and experiment scripts.

## Quick start

```python
import numpy as np
from prony.core import prony_method
from prony.data import generate_clean_data
from prony.utils import match_estimates

n = 2
t = np.arange(30, dtype=float)
a_true = np.array([1.0, 0.5])
omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j])
y = generate_clean_data(t, a_true, omega_true)

a_est, omega_est, hankel_condition = prony_method(
    y, oversampling_factor=2, n=n
)
a_est, omega_est, distances = match_estimates(
    a_true, omega_true, a_est, omega_est
)
```

## Reproducing the paired oversampling study

The complete design, noise protocol, metric definitions, runner sequence,
interpretation limits, and output map are documented in
[`experiments/paired_oversampling/README.md`](experiments/paired_oversampling/README.md).

The curated results are available immediately under
[`reproducibility/paired_oversampling/`](reproducibility/paired_oversampling/README.md).
They include all 14 NPZ datasets, 50 CSV files, and 18 PNG figures present in
the finalized local results snapshot. These are archived outputs, not files
that are automatically regenerated during installation or test execution.

To regenerate outputs, run the experiment commands documented in the paired
study README. Generated working outputs are written under `results/`, which is
intentionally ignored by Git so newly generated or intermediate files are not
published accidentally. The curated snapshot under `reproducibility/` is
version-controlled.

## Interpretation and scope

The design varies model order `n=4`, oversampling factor `rho=1,...,10`, and
ten noise levels from 0 to 0.5, with 1000 paired Monte Carlo trials per
configuration. Real Gaussian noise is the primary protocol; circular complex
Gaussian noise is an audit. The sample count and observation horizon increase
with `rho`, so the results characterize the complete oversampling design and
do not isolate Hankel aspect ratio as a causal mechanism.

The final parameter summaries use wrap-aware exponent matching and the
corresponding wrapped-match amplitude metric. A lower condition number of the
full noisy Hankel matrix is not, by itself, evidence of improved ESPRIT
parameter recovery. The observed favourable oversampling range depends on
noise level, signal structure, and the quantity being estimated; exact grid
minima should not be interpreted as a universal or uniquely optimal choice.

## Project layout

```text
src/prony/                         reusable implementation
experiments/paired_oversampling/  research runners and documentation
reproducibility/paired_oversampling/
  data/                            trial-level NPZ datasets
  tables/summaries/                summary and diagnostic CSV files
  tables/formatted/                formatted CSV tables
  figures/                         PNG figures
tests/                             automated tests
```

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
