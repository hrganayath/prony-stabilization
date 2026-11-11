"""Bootstrap uncertainty for the second-frequency damping sweep.

This script does not rerun Monte Carlo simulations.

Input:
    results/paired_oversampling/damping_sweep_closefreq.npz

Outputs:
    results/paired_oversampling/damping_sweep_closefreq_best_rho_bootstrap.csv

The bootstrap preserves pairing across rho within each trial.

Run:
    python3.10 -m experiments.paired_oversampling.damping_sweep_closefreq_uncertainty
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "results" / "paired_oversampling"

INPUT = RESULT_DIR / "damping_sweep_closefreq.npz"
OUTPUT = RESULT_DIR / "damping_sweep_closefreq_best_rho_bootstrap.csv"

PRIMARY_METRICS = (
    "wrapped_exponent_error",
    "relative_amplitude_error",
)

N_BOOT = 10000
BOOT_SEED = 20261008


def load_data():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}")

    data = np.load(INPUT, allow_pickle=False)

    raw = data["raw"]
    metric_names = [str(x) for x in data["metric_names"]]
    damping_scales = np.asarray(data["damping_scales"], dtype=float)
    noise_levels = np.asarray(data["noise_levels"], dtype=float)
    rho_values = np.asarray(data["rho_values"], dtype=int)
    trial_seeds = np.asarray(data["trial_seeds"], dtype=int)

    if raw.shape[3] != len(trial_seeds):
        raise ValueError("Trial dimension mismatch.")

    return (
        raw,
        metric_names,
        damping_scales,
        noise_levels,
        rho_values,
        trial_seeds,
    )


def bootstrap_best_rho():
    (
        raw,
        metric_names,
        damping_scales,
        noise_levels,
        rho_values,
        trial_seeds,
    ) = load_data()

    metric_index = {name: i for i, name in enumerate(metric_names)}
    rng = np.random.default_rng(BOOT_SEED)
    n_trials = len(trial_seeds)

    rows = []

    for metric in PRIMARY_METRICS:
        if metric not in metric_index:
            raise ValueError(f"Missing metric: {metric}")

        mi = metric_index[metric]

        for ai, alpha in enumerate(damping_scales):
            for si, sigma in enumerate(noise_levels):
                block = raw[ai, si, :, :, mi]  # rho x trial

                original_means = np.mean(block, axis=1)
                original_best_rho = int(
                    rho_values[int(np.argmin(original_means))]
                )

                counts = np.zeros(len(rho_values), dtype=int)

                remaining = N_BOOT
                batch_size = 250

                while remaining > 0:
                    b = min(batch_size, remaining)

                    indices = rng.integers(
                        0,
                        n_trials,
                        size=(b, n_trials),
                    )

                    # Pairing across rho is preserved by using the same
                    # resampled trial indices for every rho.
                    sampled = block[:, indices]   # rho x boot x trial
                    means = sampled.mean(axis=2)  # rho x boot
                    best_idx = np.argmin(means, axis=0)

                    counts += np.bincount(
                        best_idx,
                        minlength=len(rho_values),
                    )

                    remaining -= b

                for ri, rho in enumerate(rho_values):
                    rows.append(
                        {
                            "damping_scale": float(alpha),
                            "noise_level": float(sigma),
                            "metric": metric,
                            "original_best_rho": original_best_rho,
                            "rho": int(rho),
                            "bootstrap_best_rho_probability": float(
                                counts[ri] / N_BOOT
                            ),
                            "bootstrap_best_rho_count": int(counts[ri]),
                        }
                    )

    return pd.DataFrame(rows)


def main():
    out = bootstrap_best_rho()
    out.to_csv(OUTPUT, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 6)

    print()
    print("=" * 105)
    print("SECOND-FREQUENCY DAMPING SWEEP: BOOTSTRAP BEST-RHO PROBABILITIES >= 5%")
    print("=" * 105)

    shown = out[out["bootstrap_best_rho_probability"] >= 0.05].copy()

    print(
        shown[
            [
                "damping_scale",
                "noise_level",
                "metric",
                "original_best_rho",
                "rho",
                "bootstrap_best_rho_probability",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Saved bootstrap results to: {OUTPUT}")
    print(f"Bootstrap replicates: {N_BOOT}")
    print(f"Bootstrap seed: {BOOT_SEED}")


if __name__ == "__main__":
    main()
