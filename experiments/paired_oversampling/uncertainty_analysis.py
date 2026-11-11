"""Paired uncertainty analysis for the CMMAI oversampling study.

Purpose
-------
Use the already-generated 1000 paired trials from audit_real.npz to quantify:

1. Paired mean differences relative to rho=1 with 95% bootstrap CIs.
2. Near-optimal rho sets based on the mean curve.
3. Bootstrap uncertainty in the grid-minimizing rho.

Primary metrics
---------------
- wrapped_exponent_error
- wrapped_match_amplitude_error

No new Monte Carlo simulation is performed. This script only resamples the
existing paired trials.

Run:
    python3.10 -m experiments.paired_oversampling.uncertainty_analysis
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "results" / "paired_oversampling"

AUDIT_PATH = RESULT_DIR / "audit_real.npz"

OUTPUT_DIFFS = RESULT_DIR / "uncertainty_paired_differences.csv"
OUTPUT_NEAR = RESULT_DIR / "uncertainty_near_optimal_sets.csv"
OUTPUT_BEST = RESULT_DIR / "uncertainty_best_rho_bootstrap.csv"

PRIMARY_METRICS = (
    "wrapped_exponent_error",
    "wrapped_match_amplitude_error",
)

N_BOOT = 10000
BOOT_SEED = 20261006
NEAR_OPTIMAL_TOL = 0.05  # within 5% of minimum mean


def load_audit():
    if not AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {AUDIT_PATH}. Run the real-noise audit first."
        )

    data = np.load(AUDIT_PATH, allow_pickle=False)
    raw = data["raw"]
    metric_names = [str(x) for x in data["metric_names"]]
    noise_levels = data["noise_levels"].astype(float)
    rho_values = data["rho_values"].astype(int)
    trial_seeds = data["trial_seeds"].astype(int)

    if raw.shape[2] != len(trial_seeds):
        raise ValueError("Trial dimension does not match saved trial_seeds.")

    return raw, metric_names, noise_levels, rho_values, trial_seeds


def percentile_ci(values: np.ndarray, alpha: float = 0.05):
    lo = float(np.quantile(values, alpha / 2.0))
    hi = float(np.quantile(values, 1.0 - alpha / 2.0))
    return lo, hi


def analyze_paired_differences(
    raw,
    metric_names,
    noise_levels,
    rho_values,
    rng,
):
    rows = []
    metric_index = {name: i for i, name in enumerate(metric_names)}
    n_trials = raw.shape[2]

    for metric in PRIMARY_METRICS:
        mi = metric_index[metric]

        for si, sigma in enumerate(noise_levels):
            baseline = raw[si, 0, :, mi]

            for ri, rho in enumerate(rho_values):
                current = raw[si, ri, :, mi]
                diff = current - baseline

                mean_current = float(np.mean(current))
                mean_baseline = float(np.mean(baseline))
                mean_diff = float(np.mean(diff))

                if rho == 1:
                    ci_lo = ci_hi = 0.0
                    prob_improve = 0.5
                else:
                    indices = rng.integers(
                        0,
                        n_trials,
                        size=(N_BOOT, n_trials),
                    )
                    boot_diff = np.mean(diff[indices], axis=1)
                    ci_lo, ci_hi = percentile_ci(boot_diff)
                    prob_improve = float(np.mean(boot_diff < 0.0))

                improvement_pct = (
                    100.0 * (mean_baseline - mean_current) / mean_baseline
                )

                rows.append(
                    {
                        "noise_level": float(sigma),
                        "metric": metric,
                        "rho": int(rho),
                        "rho1_mean": mean_baseline,
                        "rho_mean": mean_current,
                        "paired_mean_difference_rho_minus_rho1": mean_diff,
                        "ci95_low": ci_lo,
                        "ci95_high": ci_hi,
                        "improvement_vs_rho1_pct": float(improvement_pct),
                        "bootstrap_probability_improvement": prob_improve,
                    }
                )

    return pd.DataFrame(rows)


def analyze_near_optimal_sets(
    raw,
    metric_names,
    noise_levels,
    rho_values,
):
    rows = []
    metric_index = {name: i for i, name in enumerate(metric_names)}

    for metric in PRIMARY_METRICS:
        mi = metric_index[metric]

        for si, sigma in enumerate(noise_levels):
            means = np.mean(raw[si, :, :, mi], axis=1)
            best_idx = int(np.argmin(means))
            best_rho = int(rho_values[best_idx])
            best_mean = float(means[best_idx])

            threshold = best_mean * (1.0 + NEAR_OPTIMAL_TOL)
            near_mask = means <= threshold
            near_rhos = [int(x) for x in rho_values[near_mask]]

            rows.append(
                {
                    "noise_level": float(sigma),
                    "metric": metric,
                    "best_rho": best_rho,
                    "best_mean": best_mean,
                    "tolerance_fraction": NEAR_OPTIMAL_TOL,
                    "near_optimal_rhos": ",".join(map(str, near_rhos)),
                    "near_optimal_min_rho": min(near_rhos),
                    "near_optimal_max_rho": max(near_rhos),
                    "near_optimal_count": len(near_rhos),
                }
            )

    return pd.DataFrame(rows)


def analyze_best_rho_bootstrap(
    raw,
    metric_names,
    noise_levels,
    rho_values,
    rng,
):
    rows = []
    metric_index = {name: i for i, name in enumerate(metric_names)}
    n_trials = raw.shape[2]

    for metric in PRIMARY_METRICS:
        mi = metric_index[metric]

        for si, sigma in enumerate(noise_levels):
            block = raw[si, :, :, mi]  # shape: rho x trial

            original_means = np.mean(block, axis=1)
            original_best_rho = int(rho_values[np.argmin(original_means)])

            counts = {int(rho): 0 for rho in rho_values}

            batch = 250
            remaining = N_BOOT

            while remaining > 0:
                this_batch = min(batch, remaining)
                indices = rng.integers(
                    0,
                    n_trials,
                    size=(this_batch, n_trials),
                )

                # Preserve pairing: every rho uses the same resampled trial
                # indices within each bootstrap replicate.
                sampled = block[:, indices]  # rho x bootstrap x trial
                means = sampled.mean(axis=2)  # rho x bootstrap
                best_indices = np.argmin(means, axis=0)

                for idx in best_indices:
                    counts[int(rho_values[int(idx)])] += 1

                remaining -= this_batch

            total = float(N_BOOT)
            for rho in rho_values:
                prob = counts[int(rho)] / total
                rows.append(
                    {
                        "noise_level": float(sigma),
                        "metric": metric,
                        "original_best_rho": original_best_rho,
                        "rho": int(rho),
                        "bootstrap_best_rho_probability": float(prob),
                        "bootstrap_best_rho_count": int(counts[int(rho)]),
                    }
                )

    return pd.DataFrame(rows)


def print_compact_summary(near_df, best_df):
    print()
    print("NEAR-OPTIMAL SETS (within 5% of minimum mean)")
    print(
        near_df[
            [
                "noise_level",
                "metric",
                "best_rho",
                "near_optimal_rhos",
            ]
        ].to_string(index=False)
    )

    print()
    print("BOOTSTRAP BEST-RHO PROBABILITIES >= 5%")
    show = best_df[
        best_df["bootstrap_best_rho_probability"] >= 0.05
    ].copy()

    print(
        show[
            [
                "noise_level",
                "metric",
                "original_best_rho",
                "rho",
                "bootstrap_best_rho_probability",
            ]
        ].to_string(index=False)
    )


def main():
    raw, metric_names, noise_levels, rho_values, trial_seeds = load_audit()

    missing = [m for m in PRIMARY_METRICS if m not in metric_names]
    if missing:
        raise ValueError(
            f"Required metrics missing from audit_real.npz: {missing}"
        )

    rng = np.random.default_rng(BOOT_SEED)

    diffs = analyze_paired_differences(
        raw,
        metric_names,
        noise_levels,
        rho_values,
        rng,
    )

    near = analyze_near_optimal_sets(
        raw,
        metric_names,
        noise_levels,
        rho_values,
    )

    best = analyze_best_rho_bootstrap(
        raw,
        metric_names,
        noise_levels,
        rho_values,
        rng,
    )

    diffs.to_csv(OUTPUT_DIFFS, index=False)
    near.to_csv(OUTPUT_NEAR, index=False)
    best.to_csv(OUTPUT_BEST, index=False)

    print(f"Saved paired-difference uncertainty to: {OUTPUT_DIFFS}")
    print(f"Saved near-optimal sets to:            {OUTPUT_NEAR}")
    print(f"Saved best-rho bootstrap to:           {OUTPUT_BEST}")
    print(f"Bootstrap replicates:                  {N_BOOT}")
    print(f"Bootstrap seed:                        {BOOT_SEED}")

    print_compact_summary(near, best)


if __name__ == "__main__":
    main()
