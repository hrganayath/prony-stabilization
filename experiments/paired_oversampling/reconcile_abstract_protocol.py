"""Reproduce the pre-cleanup paired protocol used by the older experiment runner.

The older experiments/run_experiments.py reset np.random.default_rng(trial)
inside every (rho, sigma) configuration. Therefore trial i reused the same
standard-normal stream across rho and sigma, with longer rho values taking
longer prefixes, but the generator was PCG64/default_rng rather than
legacy RandomState.

This script exists only to reconcile the accepted-abstract numbers with the
current reproducible baseline. It uses the current estimator and historical
(unwrapped) exponent metric/pole-plane matching.

Run:
    python -m experiments.paired_oversampling.reconcile_abstract_protocol
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from prony import match_estimates, prony_method, relative_exponent_error

from .config import (
    AMPLITUDES,
    EXPONENTS,
    MAX_SAMPLES,
    MODEL_ORDER,
    N_TRIALS,
    NOISE_LEVELS,
    RHO_VALUES,
    clean_signal,
    sample_count,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "paired_oversampling"


def run() -> tuple[np.ndarray, np.ndarray]:
    y_clean = clean_signal(MAX_SAMPLES)

    exp_err = np.empty(
        (len(NOISE_LEVELS), len(RHO_VALUES), N_TRIALS),
        dtype=float,
    )
    amp_err = np.empty_like(exp_err)

    for noise_idx, sigma in enumerate(NOISE_LEVELS):
        print(
            f"legacy-default_rng: noise {noise_idx + 1}/{len(NOISE_LEVELS)} "
            f"sigma={sigma:.6f}",
            flush=True,
        )

        for trial in range(N_TRIALS):
            # This exactly mirrors the old experiment's per-trial generator:
            # np.random.default_rng(trial), reset for every configuration.
            rng = np.random.default_rng(trial)
            noise_full = rng.standard_normal(MAX_SAMPLES)

            for rho_idx, rho in enumerate(RHO_VALUES):
                n_samples = sample_count(int(rho))
                y_noisy = (
                    y_clean[:n_samples]
                    + float(sigma) * noise_full[:n_samples]
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    a_hat, lambda_hat, _ = prony_method(
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

                exp_err[noise_idx, rho_idx, trial] = relative_exponent_error(
                    EXPONENTS,
                    lambda_hat,
                )
                amp_err[noise_idx, rho_idx, trial] = (
                    np.linalg.norm(a_hat - AMPLITUDES)
                    / np.linalg.norm(AMPLITUDES)
                )

    return exp_err, amp_err


def build_minima(exp_err: np.ndarray, amp_err: np.ndarray) -> pd.DataFrame:
    rows = []

    for i, sigma in enumerate(NOISE_LEVELS):
        exp_means = exp_err[i].mean(axis=1)
        amp_means = amp_err[i].mean(axis=1)

        j_exp = int(np.argmin(exp_means))
        j_amp = int(np.argmin(amp_means))

        rows.append(
            {
                "sigma": float(sigma),
                "best_rho_exponent": int(RHO_VALUES[j_exp]),
                "rho1_exponent_mean": float(exp_means[0]),
                "best_exponent_mean": float(exp_means[j_exp]),
                "exponent_gain_vs_rho1_pct": float(
                    100.0 * (exp_means[0] - exp_means[j_exp]) / exp_means[0]
                ) if exp_means[0] > 0 else 0.0,
                "best_rho_amplitude": int(RHO_VALUES[j_amp]),
                "rho1_amplitude_mean": float(amp_means[0]),
                "best_amplitude_mean": float(amp_means[j_amp]),
                "amplitude_gain_vs_rho1_pct": float(
                    100.0 * (amp_means[0] - amp_means[j_amp]) / amp_means[0]
                ) if amp_means[0] > 0 else 0.0,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    exp_err, amp_err = run()
    minima = build_minima(exp_err, amp_err)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT_DIR / "abstract_protocol_default_rng.npz"
    csv_path = OUTPUT_DIR / "abstract_protocol_default_rng_minima.csv"

    np.savez_compressed(
        raw_path,
        exponent_error=exp_err,
        amplitude_error=amp_err,
        noise_levels=NOISE_LEVELS,
        rho_values=RHO_VALUES,
    )
    minima.to_csv(csv_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.precision", 6)

    print()
    print("LEGACY default_rng MINIMA")
    print(minima.to_string(index=False))
    print()
    print(f"Saved raw:    {raw_path}")
    print(f"Saved minima: {csv_path}")


if __name__ == "__main__":
    main()
