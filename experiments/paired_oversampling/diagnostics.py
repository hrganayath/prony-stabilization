"""Mechanism diagnostics for the paired oversampling study.

This module deliberately leaves the baseline experiment in run.py unchanged.
It reruns the same paired noise protocol and records additional diagnostics
that help interpret the observed oversampling trade-off.

Run a quick smoke test from the repository root with:

    python -m experiments.paired_oversampling.diagnostics --trials 3

Run the full 1000-trial diagnostic study with:

    python -m experiments.paired_oversampling.diagnostics
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import hankel, subspace_angles, svd

from prony import match_estimates, prony_method, wrap_to_nyquist

from .config import (
    AMPLITUDES,
    EXPONENTS,
    FREQUENCIES,
    MAX_SAMPLES,
    MODEL_ORDER,
    N_TRIALS,
    NOISE_LEVELS,
    RHO_VALUES,
    TRIAL_SEEDS,
    clean_signal,
    sample_count,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "paired_oversampling"

DIAGNOSTIC_NAMES = (
    "damping_rmse",
    "frequency_rmse",
    "oracle_amplitude_error",
    "subspace_angle_max_deg",
    "subspace_angle_mean_deg",
)


def _build_hankel(y: np.ndarray, rho: int) -> np.ndarray:
    """Construct the same Hankel matrix used by prony_method."""
    m = int(rho) * MODEL_ORDER
    return hankel(y[: m + 1], y[m : m + MODEL_ORDER + 1])


def _oracle_amplitudes(
    y_noisy: np.ndarray,
    length: int,
) -> np.ndarray:
    """Estimate amplitudes with the true exponents held fixed.

    The solve mirrors the column-scaled Vandermonde least-squares stage in
    prony_method. It therefore isolates amplitude estimation from pole
    estimation error.
    """
    k = np.arange(length, dtype=float)
    V = np.exp(np.outer(k, EXPONENTS))

    col_norms = np.linalg.norm(V, axis=0)
    threshold = np.finfo(float).eps * np.max(col_norms)
    col_norms[col_norms < threshold] = 1.0

    V_scaled = V / col_norms
    a_scaled, _, _, _ = np.linalg.lstsq(
        V_scaled,
        y_noisy[:length],
        rcond=None,
    )
    return a_scaled / col_norms


def _parameter_diagnostics(
    lambda_hat: np.ndarray,
) -> tuple[float, float]:
    """Return damping RMSE and wrapped frequency RMSE."""
    damping_error = np.real(lambda_hat) - np.real(EXPONENTS)
    damping_rmse = float(
        np.sqrt(np.mean(damping_error**2))
    )

    f_hat = np.imag(lambda_hat) / (2.0 * np.pi)
    frequency_error = wrap_to_nyquist(f_hat - FREQUENCIES)
    frequency_rmse = float(
        np.sqrt(np.mean(frequency_error**2))
    )

    return damping_rmse, frequency_rmse


def _subspace_diagnostics(
    y_clean: np.ndarray,
    y_noisy: np.ndarray,
    rho: int,
) -> tuple[float, float]:
    """Compare noisy and clean rank-n left singular subspaces.

    Returns the largest and mean principal angles in degrees.
    """
    H_clean = _build_hankel(y_clean, rho)
    H_noisy = _build_hankel(y_noisy, rho)

    U_clean, _, _ = svd(H_clean, full_matrices=False)
    U_noisy, _, _ = svd(H_noisy, full_matrices=False)

    Uc = U_clean[:, :MODEL_ORDER]
    Un = U_noisy[:, :MODEL_ORDER]

    angles = np.rad2deg(subspace_angles(Uc, Un))

    return float(np.max(angles)), float(np.mean(angles))


def _one_trial(
    y_clean: np.ndarray,
    y_noisy: np.ndarray,
    rho: int,
) -> np.ndarray:
    """Compute the additional diagnostics for one paired fit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a_hat, lambda_hat, _ = prony_method(
            y_noisy,
            oversampling_factor=int(rho),
            n=MODEL_ORDER,
        )

    _, lambda_hat, _ = match_estimates(
        AMPLITUDES,
        EXPONENTS,
        a_hat,
        lambda_hat,
    )

    damping_rmse, frequency_rmse = _parameter_diagnostics(lambda_hat)

    n_samples = sample_count(int(rho))
    a_oracle = _oracle_amplitudes(y_noisy, n_samples)
    oracle_amplitude_error = float(
        np.linalg.norm(a_oracle - AMPLITUDES)
        / np.linalg.norm(AMPLITUDES)
    )

    angle_max, angle_mean = _subspace_diagnostics(
        y_clean=y_clean[:n_samples],
        y_noisy=y_noisy,
        rho=int(rho),
    )

    values = np.array(
        [
            damping_rmse,
            frequency_rmse,
            oracle_amplitude_error,
            angle_max,
            angle_mean,
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(values)):
        raise FloatingPointError(
            f"Non-finite diagnostic for rho={rho}: {values}"
        )

    return values


def run(n_trials: int = N_TRIALS) -> np.ndarray:
    """Run the diagnostic study under the same paired-noise protocol."""
    if not 1 <= n_trials <= N_TRIALS:
        raise ValueError(
            f"n_trials must be between 1 and {N_TRIALS}, got {n_trials}."
        )

    y_clean_full = clean_signal(MAX_SAMPLES)
    seeds = TRIAL_SEEDS[:n_trials]

    raw = np.empty(
        (
            len(NOISE_LEVELS),
            len(RHO_VALUES),
            n_trials,
            len(DIAGNOSTIC_NAMES),
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
            rng = np.random.RandomState(int(seed))
            noise_full = rng.standard_normal(MAX_SAMPLES)

            for rho_idx, rho in enumerate(RHO_VALUES):
                n_samples = sample_count(int(rho))
                y_noisy = (
                    y_clean_full[:n_samples]
                    + float(sigma) * noise_full[:n_samples]
                )

                raw[noise_idx, rho_idx, trial_idx, :] = _one_trial(
                    y_clean=y_clean_full,
                    y_noisy=y_noisy,
                    rho=int(rho),
                )

    return raw


def make_summary(raw: np.ndarray) -> pd.DataFrame:
    """Create descriptive summaries from raw diagnostic values."""
    rows: list[dict[str, float | int | str]] = []

    for noise_idx, sigma in enumerate(NOISE_LEVELS):
        for rho_idx, rho in enumerate(RHO_VALUES):
            block = raw[noise_idx, rho_idx, :, :]

            for diagnostic_idx, name in enumerate(DIAGNOSTIC_NAMES):
                values = block[:, diagnostic_idx]
                rows.append(
                    {
                        "noise_level": float(sigma),
                        "rho": int(rho),
                        "diagnostic": name,
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "median": float(np.median(values)),
                        "q25": float(np.quantile(values, 0.25)),
                        "q75": float(np.quantile(values, 0.75)),
                        "q95": float(np.quantile(values, 0.95)),
                    }
                )

    return pd.DataFrame(rows)


def save_results(
    raw: np.ndarray,
    n_trials: int,
) -> tuple[Path, Path]:
    """Save raw diagnostics and their summaries."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if n_trials == N_TRIALS:
        raw_path = OUTPUT_DIR / "diagnostics_raw.npz"
        summary_path = OUTPUT_DIR / "diagnostics_summary.csv"
    else:
        raw_path = OUTPUT_DIR / f"diagnostics_raw_smoke_{n_trials}.npz"
        summary_path = OUTPUT_DIR / f"diagnostics_summary_smoke_{n_trials}.csv"

    np.savez_compressed(
        raw_path,
        raw=raw,
        diagnostic_names=np.array(DIAGNOSTIC_NAMES),
        noise_levels=NOISE_LEVELS,
        rho_values=RHO_VALUES,
        trial_seeds=TRIAL_SEEDS[:n_trials],
    )

    summary = make_summary(raw)
    summary.to_csv(summary_path, index=False)

    return raw_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired oversampling mechanism diagnostics."
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
    raw_path, summary_path = save_results(
        raw,
        n_trials=args.trials,
    )

    print()
    print(f"Saved raw diagnostics to: {raw_path}")
    print(f"Saved summaries to:       {summary_path}")
    print(f"Raw array shape:           {raw.shape}")


if __name__ == "__main__":
    main()
