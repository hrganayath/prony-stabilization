"""Paired oversampling Monte Carlo study.

Run from the repository root with:

    python -m experiments.paired_oversampling.run

For a quick smoke test:

    python -m experiments.paired_oversampling.run --trials 3

This runner is intentionally separate from experiments.run_experiments.
Its purpose is to reproduce the accepted-abstract paired oversampling design:
trial i uses default_rng(i), and all oversampling factors reuse progressively
longer prefixes of the same realization.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from prony import match_estimates, prony_method, relative_exponent_error

from .noise_protocol import canonical_real_noise\n\nfrom .config import (
    AMPLITUDES,
    EXPONENTS,
    MAX_SAMPLES,
    MODEL_ORDER,
    N_TRIALS,
    NOISE_LEVELS,
    RHO_VALUES,
    SCORING_LENGTH,
    TRIAL_SEEDS,
    clean_signal,
    sample_count,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "paired_oversampling"
RAW_RESULTS = OUTPUT_DIR / "raw_results.npz"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"

METRIC_NAMES = (
    "exponent_error",
    "amplitude_error",
    "rmse_noisy",
    "rmse_clean",
    "hankel_condition",
)


def _reconstruct(
    amplitudes: np.ndarray,
    exponents: np.ndarray,
    length: int,
) -> np.ndarray:
    """Reconstruct a sum of exponentials on k=0,...,length-1."""
    k = np.arange(length, dtype=float)
    V = np.exp(np.outer(k, exponents))
    return V @ amplitudes


def _trial_metrics(
    y_noisy: np.ndarray,
    rho: int,
    y_clean_reference: np.ndarray,
) -> np.ndarray:
    """Run one fit and return the five paper-level metrics."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a_hat, lambda_hat, cond_h = prony_method(
            y_noisy,
            oversampling_factor=int(rho),
            n=MODEL_ORDER,
        )

    a_hat, lambda_hat, _ = match_estimates(
        AMPLITUDES,
        EXPONENTS,
        a_hat,
        lambda_hat,
    )

    exponent_error = relative_exponent_error(EXPONENTS, lambda_hat)
    amplitude_error = (
        np.linalg.norm(a_hat - AMPLITUDES)
        / np.linalg.norm(AMPLITUDES)
    )

    y_hat = _reconstruct(a_hat, lambda_hat, SCORING_LENGTH)
    y_noisy_eval = y_noisy[:SCORING_LENGTH]
    y_clean_eval = y_clean_reference[:SCORING_LENGTH]

    rmse_noisy = np.linalg.norm(y_hat - y_noisy_eval) / np.sqrt(SCORING_LENGTH)
    rmse_clean = np.linalg.norm(y_hat - y_clean_eval) / np.sqrt(SCORING_LENGTH)

    metrics = np.array(
        [
            exponent_error,
            amplitude_error,
            rmse_noisy,
            rmse_clean,
            cond_h,
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(metrics)):
        raise FloatingPointError(
            f"Non-finite metric for rho={rho}: {metrics}"
        )

    return metrics


def run(n_trials: int = N_TRIALS) -> np.ndarray:
    """Run the paired Monte Carlo experiment.

    Parameters
    ----------
    n_trials : int
        Number of paired trials per noise level. The paper run uses 1000.
        Smaller values are useful only for smoke testing.

    Returns
    -------
    raw : np.ndarray
        Shape
        (n_noise_levels, n_rho_values, n_trials, n_metrics).
    """
    if not 1 <= n_trials <= N_TRIALS:
        raise ValueError(
            f"n_trials must be between 1 and {N_TRIALS}, got {n_trials}."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_clean_full = clean_signal(MAX_SAMPLES)
    seeds = TRIAL_SEEDS[:n_trials]

    raw = np.empty(
        (
            len(NOISE_LEVELS),
            len(RHO_VALUES),
            n_trials,
            len(METRIC_NAMES),
        ),
        dtype=float,
    )

    for noise_idx, sigma in enumerate(NOISE_LEVELS):
        print(
            f"Noise {noise_idx + 1}/{len(NOISE_LEVELS)}: "
            f"sigma={sigma:.6f}",
            flush=True,
        )

        for trial_idx, seed in enumerate(seeds):
            # Accepted-abstract paired protocol:
            # trial i uses default_rng(i). The same standard-normal
            # realization is reused across rho via progressively longer
            # prefixes. Reusing the same seed at every sigma also keeps the
            # underlying standard-normal draw fixed across noise levels.
            noise_full = canonical_real_noise(int(seed), MAX_SAMPLES)

            for rho_idx, rho in enumerate(RHO_VALUES):
                n_samples = sample_count(int(rho))
                y_noisy = (
                    y_clean_full[:n_samples]
                    + float(sigma) * noise_full[:n_samples]
                )

                raw[noise_idx, rho_idx, trial_idx, :] = _trial_metrics(
                    y_noisy=y_noisy,
                    rho=int(rho),
                    y_clean_reference=y_clean_full,
                )

    return raw


def save_results(raw: np.ndarray, n_trials: int) -> tuple[Path, Path]:
    """Save raw and summarized results.

    Smoke-test outputs are kept separate from the full 1000-trial outputs.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if n_trials == N_TRIALS:
        raw_path = RAW_RESULTS
        summary_path = SUMMARY_CSV
    else:
        raw_path = OUTPUT_DIR / f"raw_results_smoke_{n_trials}.npz"
        summary_path = OUTPUT_DIR / f"summary_smoke_{n_trials}.csv"

    np.savez_compressed(
        raw_path,
        raw=raw,
        metric_names=np.array(METRIC_NAMES),
        noise_levels=NOISE_LEVELS,
        rho_values=RHO_VALUES,
        trial_seeds=TRIAL_SEEDS[:n_trials],
    )

    summary = make_summary(raw)
    summary.to_csv(summary_path, index=False)

    return raw_path, summary_path


def make_summary(raw: np.ndarray) -> pd.DataFrame:
    """Create descriptive summaries from raw trial-level results."""
    rows: list[dict[str, float | int | str]] = []

    for noise_idx, sigma in enumerate(NOISE_LEVELS):
        for rho_idx, rho in enumerate(RHO_VALUES):
            block = raw[noise_idx, rho_idx, :, :]

            for metric_idx, metric_name in enumerate(METRIC_NAMES):
                values = block[:, metric_idx]
                rows.append(
                    {
                        "noise_level": float(sigma),
                        "rho": int(rho),
                        "metric": metric_name,
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "median": float(np.median(values)),
                        "q25": float(np.quantile(values, 0.25)),
                        "q75": float(np.quantile(values, 0.75)),
                        "q95": float(np.quantile(values, 0.95)),
                    }
                )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paired oversampling Monte Carlo study."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=N_TRIALS,
        help=(
            f"Number of paired trials per noise level "
            f"(default: {N_TRIALS}). Use a small value for a smoke test."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw = run(n_trials=args.trials)
    raw_path, summary_path = save_results(raw, n_trials=args.trials)

    print()
    print(f"Saved raw trial data to: {raw_path}")
    print(f"Saved summaries to:      {summary_path}")
    print(f"Raw array shape:          {raw.shape}")


if __name__ == "__main__":
    main()
