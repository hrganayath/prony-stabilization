"""Uncertainty analysis for the new CMMAI control experiments.

This script does not rerun any Monte Carlo simulations.

It reads the existing raw trial files from:
    results/paired_oversampling/

Fixed-N shape control:
    fixed_n_shape_N17.npz
    fixed_n_shape_N29.npz
    fixed_n_shape_N45.npz

Damping sweep:
    damping_sweep.npz

It produces:
1. Paired bootstrap 95% CIs for fixed-N shape comparisons:
   - original narrow C=5 versus the best tested C
   - original narrow C=5 versus the near-balanced C
2. Bootstrap uncertainty for the grid-minimizing rho in the damping sweep.
3. Median and tail summaries for the fixed-N shape control at sigma=0.5.

No new noisy signals are generated.

Run:
    python3.10 -m experiments.paired_oversampling.controls_uncertainty
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "results" / "paired_oversampling"

N_VALUES = (17, 29, 45)
TARGET_NOISE_LEVELS = (2.0 / 9.0, 0.5)

PRIMARY_METRICS = (
    "wrapped_exponent_error",
    "relative_amplitude_error",
)

N_BOOT = 10000
BOOT_SEED = 20261007

OUT_SHAPE_CI = RESULT_DIR / "fixed_n_shape_bootstrap_ci.csv"
OUT_SHAPE_TAILS = RESULT_DIR / "fixed_n_shape_high_noise_tails.csv"
OUT_DAMPING = RESULT_DIR / "damping_sweep_best_rho_bootstrap.csv"


def percentile_ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    return (
        float(np.quantile(x, alpha / 2.0)),
        float(np.quantile(x, 1.0 - alpha / 2.0)),
    )


def load_fixed_n(N: int):
    path = RESULT_DIR / f"fixed_n_shape_N{N}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    data = np.load(path, allow_pickle=False)

    raw = data["raw"]
    metric_names = [str(x) for x in data["metric_names"]]
    noise_levels = np.asarray(data["noise_levels"], dtype=float)
    columns = np.asarray(data["columns"], dtype=int)
    trial_seeds = np.asarray(data["trial_seeds"], dtype=int)

    if raw.shape[2] != len(trial_seeds):
        raise ValueError(f"Trial dimension mismatch in {path.name}")

    return raw, metric_names, noise_levels, columns, trial_seeds


def paired_bootstrap_mean_difference(
    reference: np.ndarray,
    candidate: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Bootstrap candidate-reference mean difference using paired trials."""
    if reference.shape != candidate.shape:
        raise ValueError("Paired vectors must have the same shape.")

    diff = candidate - reference
    n = len(diff)

    observed_diff = float(np.mean(diff))
    observed_gain_pct = float(
        100.0 * (np.mean(reference) - np.mean(candidate)) / np.mean(reference)
    )

    batch_size = 500
    remaining = N_BOOT
    boot_means = []

    while remaining > 0:
        b = min(batch_size, remaining)
        indices = rng.integers(0, n, size=(b, n))
        boot_means.append(np.mean(diff[indices], axis=1))
        remaining -= b

    boot_means = np.concatenate(boot_means)
    lo, hi = percentile_ci(boot_means)

    return {
        "reference_mean": float(np.mean(reference)),
        "candidate_mean": float(np.mean(candidate)),
        "paired_mean_difference_candidate_minus_reference": observed_diff,
        "ci95_low": lo,
        "ci95_high": hi,
        "gain_vs_reference_pct": observed_gain_pct,
        "bootstrap_probability_candidate_better": float(
            np.mean(boot_means < 0.0)
        ),
    }


def analyze_fixed_n(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    ci_rows = []
    tail_rows = []

    for N in N_VALUES:
        raw, metric_names, noise_levels, columns, trial_seeds = load_fixed_n(N)
        metric_index = {name: i for i, name in enumerate(metric_names)}

        if 5 not in columns:
            raise ValueError(f"C=5 missing for N={N}")

        narrow_idx = int(np.where(columns == 5)[0][0])
        balanced_idx = len(columns) - 1

        for sigma_target in TARGET_NOISE_LEVELS:
            matches = np.where(np.isclose(noise_levels, sigma_target))[0]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one match for sigma={sigma_target} in N={N}"
                )
            si = int(matches[0])

            for metric in PRIMARY_METRICS:
                if metric not in metric_index:
                    raise ValueError(f"{metric} missing in N={N}")

                mi = metric_index[metric]
                block = raw[si, :, :, mi]  # C x trial
                means = np.mean(block, axis=1)

                best_idx = int(np.argmin(means))
                best_C = int(columns[best_idx])
                balanced_C = int(columns[balanced_idx])

                narrow = block[narrow_idx]
                best = block[best_idx]
                balanced = block[balanced_idx]

                for comparison_name, candidate_C, candidate in (
                    ("best_tested_vs_narrow", best_C, best),
                    ("balanced_vs_narrow", balanced_C, balanced),
                ):
                    result = paired_bootstrap_mean_difference(
                        narrow,
                        candidate,
                        rng,
                    )
                    ci_rows.append(
                        {
                            "N": N,
                            "noise_level": float(sigma_target),
                            "metric": metric,
                            "comparison": comparison_name,
                            "reference_C": 5,
                            "candidate_C": candidate_C,
                            **result,
                        }
                    )

                if np.isclose(sigma_target, 0.5):
                    for ci, C in enumerate(columns):
                        values = block[ci]
                        tail_rows.append(
                            {
                                "N": N,
                                "noise_level": float(sigma_target),
                                "metric": metric,
                                "columns": int(C),
                                "rows": int(N - int(C) + 1),
                                "mean": float(np.mean(values)),
                                "median": float(np.median(values)),
                                "std": float(np.std(values)),
                                "q75": float(np.quantile(values, 0.75)),
                                "q90": float(np.quantile(values, 0.90)),
                                "q95": float(np.quantile(values, 0.95)),
                                "q99": float(np.quantile(values, 0.99)),
                                "max": float(np.max(values)),
                                "mean_minus_median": float(
                                    np.mean(values) - np.median(values)
                                ),
                                "mean_over_median": float(
                                    np.mean(values) / np.median(values)
                                ),
                            }
                        )

    return pd.DataFrame(ci_rows), pd.DataFrame(tail_rows)


def load_damping():
    path = RESULT_DIR / "damping_sweep.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    data = np.load(path, allow_pickle=False)

    raw = data["raw"]
    metric_names = [str(x) for x in data["metric_names"]]
    damping_scales = np.asarray(data["damping_scales"], dtype=float)
    noise_levels = np.asarray(data["noise_levels"], dtype=float)
    rho_values = np.asarray(data["rho_values"], dtype=int)
    trial_seeds = np.asarray(data["trial_seeds"], dtype=int)

    if raw.shape[3] != len(trial_seeds):
        raise ValueError("Trial dimension mismatch in damping_sweep.npz")

    return (
        raw,
        metric_names,
        damping_scales,
        noise_levels,
        rho_values,
        trial_seeds,
    )


def analyze_damping_best_rho(
    rng: np.random.Generator,
) -> pd.DataFrame:
    (
        raw,
        metric_names,
        damping_scales,
        noise_levels,
        rho_values,
        trial_seeds,
    ) = load_damping()

    metric_index = {name: i for i, name in enumerate(metric_names)}
    n_trials = len(trial_seeds)
    rows = []

    for metric in PRIMARY_METRICS:
        if metric not in metric_index:
            raise ValueError(f"{metric} missing in damping sweep")

        mi = metric_index[metric]

        for ai, alpha in enumerate(damping_scales):
            for si, sigma in enumerate(noise_levels):
                block = raw[ai, si, :, :, mi]  # rho x trial

                original_means = np.mean(block, axis=1)
                original_best_idx = int(np.argmin(original_means))
                original_best_rho = int(rho_values[original_best_idx])

                counts = np.zeros(len(rho_values), dtype=int)

                batch_size = 250
                remaining = N_BOOT

                while remaining > 0:
                    b = min(batch_size, remaining)
                    indices = rng.integers(
                        0,
                        n_trials,
                        size=(b, n_trials),
                    )

                    # Preserve pairing across rho in each bootstrap replicate.
                    sampled = block[:, indices]      # rho x boot x trial
                    means = sampled.mean(axis=2)     # rho x boot
                    best_indices = np.argmin(means, axis=0)

                    counts += np.bincount(
                        best_indices,
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


def print_shape_ci(df: pd.DataFrame) -> None:
    print()
    print("=" * 110)
    print("FIXED-N PAIRED BOOTSTRAP COMPARISONS")
    print("=" * 110)

    cols = [
        "N",
        "noise_level",
        "metric",
        "comparison",
        "reference_C",
        "candidate_C",
        "reference_mean",
        "candidate_mean",
        "paired_mean_difference_candidate_minus_reference",
        "ci95_low",
        "ci95_high",
        "gain_vs_reference_pct",
        "bootstrap_probability_candidate_better",
    ]
    print(df[cols].to_string(index=False))


def print_damping(df: pd.DataFrame) -> None:
    print()
    print("=" * 110)
    print("DAMPING-SWEEP BOOTSTRAP BEST-RHO PROBABILITIES >= 5%")
    print("=" * 110)

    shown = df[df["bootstrap_best_rho_probability"] >= 0.05].copy()

    cols = [
        "damping_scale",
        "noise_level",
        "metric",
        "original_best_rho",
        "rho",
        "bootstrap_best_rho_probability",
    ]
    print(shown[cols].to_string(index=False))


def print_high_noise_tail_summary(df: pd.DataFrame) -> None:
    print()
    print("=" * 110)
    print("HIGH-NOISE FIXED-N TAIL SUMMARY: sigma=0.5")
    print("=" * 110)

    # Print only C=5, best-mean C, and balanced C for compact inspection.
    rows = []

    for N in N_VALUES:
        for metric in PRIMARY_METRICS:
            b = df[(df["N"] == N) & (df["metric"] == metric)].sort_values(
                "columns"
            )
            best_row = b.loc[b["mean"].idxmin()]
            selected_C = {
                5,
                int(best_row["columns"]),
                int(b.iloc[-1]["columns"]),
            }
            rows.append(b[b["columns"].isin(selected_C)])

    shown = pd.concat(rows, ignore_index=True)

    cols = [
        "N",
        "metric",
        "columns",
        "mean",
        "median",
        "q90",
        "q95",
        "q99",
        "max",
        "mean_over_median",
    ]
    print(shown[cols].to_string(index=False))


def main() -> None:
    rng = np.random.default_rng(BOOT_SEED)

    shape_ci, shape_tails = analyze_fixed_n(rng)
    damping = analyze_damping_best_rho(rng)

    shape_ci.to_csv(OUT_SHAPE_CI, index=False)
    shape_tails.to_csv(OUT_SHAPE_TAILS, index=False)
    damping.to_csv(OUT_DAMPING, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.precision", 6)

    print_shape_ci(shape_ci)
    print_damping(damping)
    print_high_noise_tail_summary(shape_tails)

    print()
    print(f"Saved fixed-N bootstrap CIs to: {OUT_SHAPE_CI}")
    print(f"Saved fixed-N tail summaries to: {OUT_SHAPE_TAILS}")
    print(f"Saved damping bootstrap results to: {OUT_DAMPING}")
    print(f"Bootstrap replicates: {N_BOOT}")
    print(f"Bootstrap seed: {BOOT_SEED}")


if __name__ == "__main__":
    main()
