"""Audit runner for wrapped exponent metrics and circular-noise replication.

This module leaves the historical paired baseline in run.py unchanged.
It reruns the same rho/sigma/trial grid and records parameter estimates
plus both the historical and wrapped exponent metrics.

Examples
--------
Smoke tests:
    python -m experiments.paired_oversampling.audit_metrics --noise-model real --trials 3
    python -m experiments.paired_oversampling.audit_metrics --noise-model circular --trials 3

Full runs:
    python -m experiments.paired_oversampling.audit_metrics --noise-model real
    python -m experiments.paired_oversampling.audit_metrics --noise-model circular

Circular-noise convention:
    eps_k = sigma / sqrt(2) * (xi_k + i eta_k),
so E|eps_k|^2 = sigma^2, matching the total noise power of the
historical real-noise experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from prony import (
    match_estimates,
    match_estimates_wrapped_exponent,
    prony_method,
    relative_exponent_error,
    relative_exponent_error_wrapped,
)

from .config import (
    AMPLITUDES,
    EXPONENTS,
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

METRIC_NAMES = (
    "historical_exponent_error",
    "wrapped_exponent_error",
    "historical_amplitude_error",
    "wrapped_match_amplitude_error",
    "branch_jump_count",
    "matching_disagreement",
)


def _draw_noise(seed: int, noise_model: str) -> np.ndarray:
    """Draw one length-MAX_SAMPLES standardized noise realization."""
    rng = np.random.RandomState(int(seed))

    if noise_model == "real":
        return rng.standard_normal(MAX_SAMPLES).astype(complex)

    if noise_model == "circular":
        xi = rng.standard_normal(MAX_SAMPLES)
        eta = rng.standard_normal(MAX_SAMPLES)
        return (xi + 1j * eta) / np.sqrt(2.0)

    raise ValueError(f"Unknown noise model: {noise_model}")


def _branch_jump_count(
    omega_true: np.ndarray,
    omega_hat_matched: np.ndarray,
) -> int:
    """Count matched components whose raw imaginary difference exceeds pi."""
    raw_imag_diff = np.imag(omega_hat_matched) - np.imag(omega_true)
    return int(np.count_nonzero(np.abs(raw_imag_diff) > np.pi))


def _fit_and_score(
    y_noisy: np.ndarray,
    rho: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one trial and return metrics plus wrapped-matched estimates."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a_hat, omega_hat, _ = prony_method(
            y_noisy,
            oversampling_factor=int(rho),
            n=MODEL_ORDER,
        )

    a_hist, w_hist, _ = match_estimates(
        AMPLITUDES,
        EXPONENTS,
        a_hat,
        omega_hat,
    )
    a_wrap, w_wrap, _ = match_estimates_wrapped_exponent(
        AMPLITUDES,
        EXPONENTS,
        a_hat,
        omega_hat,
    )

    historical_exponent_error = relative_exponent_error(EXPONENTS, w_hist)
    wrapped_exponent_error = relative_exponent_error_wrapped(EXPONENTS, w_wrap)

    historical_amplitude_error = (
        np.linalg.norm(a_hist - AMPLITUDES) / np.linalg.norm(AMPLITUDES)
    )
    wrapped_match_amplitude_error = (
        np.linalg.norm(a_wrap - AMPLITUDES) / np.linalg.norm(AMPLITUDES)
    )

    branch_jump_count = _branch_jump_count(EXPONENTS, w_wrap)
    matching_disagreement = float(
        not (
            np.allclose(a_hist, a_wrap, rtol=0.0, atol=0.0)
            and np.allclose(w_hist, w_wrap, rtol=0.0, atol=0.0)
        )
    )

    metrics = np.array(
        [
            historical_exponent_error,
            wrapped_exponent_error,
            historical_amplitude_error,
            wrapped_match_amplitude_error,
            float(branch_jump_count),
            matching_disagreement,
        ],
        dtype=float,
    )

    return metrics, a_wrap, w_wrap


def run(
    noise_model: str,
    n_trials: int = N_TRIALS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the full paired audit grid for one noise model."""
    if noise_model not in {"real", "circular"}:
        raise ValueError("noise_model must be 'real' or 'circular'.")
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
            len(METRIC_NAMES),
        ),
        dtype=float,
    )
    amplitudes = np.empty(
        (
            len(NOISE_LEVELS),
            len(RHO_VALUES),
            n_trials,
            MODEL_ORDER,
        ),
        dtype=complex,
    )
    exponents = np.empty_like(amplitudes)

    for noise_idx, sigma in enumerate(NOISE_LEVELS):
        print(
            f"{noise_model}: noise {noise_idx + 1}/{len(NOISE_LEVELS)} "
            f"sigma={sigma:.6f}",
            flush=True,
        )

        for trial_idx, seed in enumerate(seeds):
            standardized_noise = _draw_noise(int(seed), noise_model)

            for rho_idx, rho in enumerate(RHO_VALUES):
                n_samples = sample_count(int(rho))
                y_noisy = (
                    y_clean_full[:n_samples]
                    + float(sigma) * standardized_noise[:n_samples]
                )

                metrics, a_hat, omega_hat = _fit_and_score(
                    y_noisy=y_noisy,
                    rho=int(rho),
                )
                raw[noise_idx, rho_idx, trial_idx, :] = metrics
                amplitudes[noise_idx, rho_idx, trial_idx, :] = a_hat
                exponents[noise_idx, rho_idx, trial_idx, :] = omega_hat

    return raw, amplitudes, exponents


def make_summary(raw: np.ndarray) -> pd.DataFrame:
    """Summarize audit metrics, including failure/jump rates."""
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


def save_results(
    raw: np.ndarray,
    amplitudes: np.ndarray,
    exponents: np.ndarray,
    noise_model: str,
    n_trials: int,
) -> tuple[Path, Path, Path]:
    """Save audit metrics, matched estimates and summary CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suffix = "" if n_trials == N_TRIALS else f"_smoke_{n_trials}"
    raw_path = OUTPUT_DIR / f"audit_{noise_model}{suffix}.npz"
    estimates_path = OUTPUT_DIR / f"estimates_{noise_model}{suffix}.npz"
    summary_path = OUTPUT_DIR / f"audit_{noise_model}_summary{suffix}.csv"

    np.savez_compressed(
        raw_path,
        raw=raw,
        metric_names=np.array(METRIC_NAMES),
        noise_levels=NOISE_LEVELS,
        rho_values=RHO_VALUES,
        trial_seeds=TRIAL_SEEDS[:n_trials],
        noise_model=np.array(noise_model),
    )
    np.savez_compressed(
        estimates_path,
        amplitudes=amplitudes,
        exponents=exponents,
        noise_levels=NOISE_LEVELS,
        rho_values=RHO_VALUES,
        trial_seeds=TRIAL_SEEDS[:n_trials],
        noise_model=np.array(noise_model),
    )

    make_summary(raw).to_csv(summary_path, index=False)

    return raw_path, estimates_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit wrapped exponent metrics and real/circular noise robustness."
        )
    )
    parser.add_argument(
        "--noise-model",
        choices=("real", "circular"),
        required=True,
        help="Noise model to run.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=N_TRIALS,
        help=f"Paired trials per noise level (default: {N_TRIALS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw, amplitudes, exponents = run(
        noise_model=args.noise_model,
        n_trials=args.trials,
    )
    raw_path, estimates_path, summary_path = save_results(
        raw=raw,
        amplitudes=amplitudes,
        exponents=exponents,
        noise_model=args.noise_model,
        n_trials=args.trials,
    )

    print()
    print(f"Saved audit metrics to: {raw_path}")
    print(f"Saved estimates to:     {estimates_path}")
    print(f"Saved summary to:       {summary_path}")
    print(f"Audit array shape:       {raw.shape}")


if __name__ == "__main__":
    main()
