"""Comparable spectral diagnostics for the paired oversampling study.

This module replaces cross-rho comparisons of left-subspace angles, whose
ambient dimension changes with rho, with diagnostics that are comparable
across rho:

1. right-subspace principal angles (fixed ambient dimension n+1),
2. clean sigma_n,
3. noisy sigma_n / sigma_(n+1) gap ratio,
4. ||H_noisy-H_clean||_2 / sigma_n(H_clean).

The historical left-angle diagnostic remains in diagnostics.py for audit
traceability but should not be used as a cross-rho headline quantity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import hankel, subspace_angles, svdvals, svd

from .config import (
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
    "right_subspace_angle_max_deg",
    "right_subspace_angle_mean_deg",
    "clean_sigma_n",
    "noisy_gap_ratio_sigma_n_over_sigma_np1",
    "noise_operator_norm",
    "noise_over_clean_sigma_n",
)


def _build_hankel(y: np.ndarray, rho: int) -> np.ndarray:
    m = int(rho) * MODEL_ORDER
    return hankel(y[: m + 1], y[m : m + MODEL_ORDER + 1])


def _one_trial(
    y_clean: np.ndarray,
    y_noisy: np.ndarray,
    rho: int,
) -> np.ndarray:
    H_clean = _build_hankel(y_clean, rho)
    H_noisy = _build_hankel(y_noisy, rho)

    _, _, Vh_clean = svd(H_clean, full_matrices=False)
    _, _, Vh_noisy = svd(H_noisy, full_matrices=False)

    # Right singular subspaces both live in C^(n+1), independent of rho.
    V_clean = Vh_clean.conj().T[:, :MODEL_ORDER]
    V_noisy = Vh_noisy.conj().T[:, :MODEL_ORDER]
    angles = np.rad2deg(subspace_angles(V_clean, V_noisy))

    s_clean = svdvals(H_clean)
    s_noisy = svdvals(H_noisy)

    sigma_n_clean = float(s_clean[MODEL_ORDER - 1])
    gap_ratio = float(s_noisy[MODEL_ORDER - 1] / s_noisy[MODEL_ORDER])

    noise_norm = float(np.linalg.norm(H_noisy - H_clean, ord=2))
    noise_over_sigma_n = float(noise_norm / sigma_n_clean)

    return np.array(
        [
            float(np.max(angles)),
            float(np.mean(angles)),
            sigma_n_clean,
            gap_ratio,
            noise_norm,
            noise_over_sigma_n,
        ],
        dtype=float,
    )


def run(n_trials: int = N_TRIALS) -> np.ndarray:
    if not 1 <= n_trials <= N_TRIALS:
        raise ValueError(
            f"n_trials must be between 1 and {N_TRIALS}, got {n_trials}."
        )

    y_clean_full = clean_signal(MAX_SAMPLES)
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
            f"spectral: noise {noise_idx + 1}/{len(NOISE_LEVELS)} "
            f"sigma={sigma:.6f}",
            flush=True,
        )
        for trial_idx, seed in enumerate(TRIAL_SEEDS[:n_trials]):
            rng = np.random.RandomState(int(seed))
            noise_full = rng.standard_normal(MAX_SAMPLES)

            for rho_idx, rho in enumerate(RHO_VALUES):
                n_samples = sample_count(int(rho))
                y_clean = y_clean_full[:n_samples]
                y_noisy = y_clean + float(sigma) * noise_full[:n_samples]

                raw[noise_idx, rho_idx, trial_idx, :] = _one_trial(
                    y_clean,
                    y_noisy,
                    int(rho),
                )

    return raw


def make_summary(raw: np.ndarray) -> pd.DataFrame:
    rows = []
    for noise_idx, sigma in enumerate(NOISE_LEVELS):
        for rho_idx, rho in enumerate(RHO_VALUES):
            for j, name in enumerate(DIAGNOSTIC_NAMES):
                values = raw[noise_idx, rho_idx, :, j]
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


def save_results(raw: np.ndarray, n_trials: int) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suffix = "" if n_trials == N_TRIALS else f"_smoke_{n_trials}"
    raw_path = OUTPUT_DIR / f"spectral_diagnostics{suffix}.npz"
    summary_path = OUTPUT_DIR / f"spectral_diagnostics_summary{suffix}.csv"

    np.savez_compressed(
        raw_path,
        raw=raw,
        diagnostic_names=np.array(DIAGNOSTIC_NAMES),
        noise_levels=NOISE_LEVELS,
        rho_values=RHO_VALUES,
        trial_seeds=TRIAL_SEEDS[:n_trials],
    )
    make_summary(raw).to_csv(summary_path, index=False)
    return raw_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-ambient and spectral oversampling diagnostics."
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
    raw = run(args.trials)
    raw_path, summary_path = save_results(raw, args.trials)
    print()
    print(f"Saved spectral diagnostics to: {raw_path}")
    print(f"Saved summary to:              {summary_path}")
    print(f"Raw array shape:               {raw.shape}")


if __name__ == "__main__":
    main()
