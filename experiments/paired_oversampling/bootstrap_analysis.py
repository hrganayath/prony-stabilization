"""Paired bootstrap confidence intervals for the canonical n=4 audit results.

This analysis uses the trial-level paired data already saved by audit_metrics.py.
No new Monte Carlo simulations are performed.

For each selected noise level, it reports paired mean differences for:
  1. rho=1 versus the mean-error minimizing moderate rho,
  2. the minimizing rho versus rho=10,
  3. the minimizing rho versus its immediate neighbors when available.

Positive diff_mean for comparison "rho_a - rho_b" means rho_b has lower error.

Run:
    python -m experiments.paired_oversampling.bootstrap_analysis
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import NOISE_LEVELS, RHO_VALUES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "paired_oversampling"

N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 20261005
SELECTED_SIGMAS = (2.0 / 9.0, 0.5)
METRICS = (
    "wrapped_exponent_error",
    "wrapped_match_amplitude_error",
)


def _load_real_audit() -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    path = OUTPUT_DIR / "audit_real.npz"
    data = np.load(path, allow_pickle=False)
    raw = data["raw"]
    names = [str(x) for x in data["metric_names"]]
    noise_levels = data["noise_levels"]
    rho_values = data["rho_values"]
    return raw, names, noise_levels, rho_values


def _paired_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Bootstrap the paired mean difference x-y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Paired samples must have the same shape.")

    d = x - y
    n = d.size
    diff_mean = float(np.mean(d))

    # Resample the paired trial differences, not x/y independently.
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = np.mean(d[indices], axis=1)

    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    p_better = float(np.mean(d > 0.0))
    return diff_mean, float(lo), float(hi), p_better


def _rho_index(rho_values: np.ndarray, rho: int) -> int:
    idx = np.where(rho_values == rho)[0]
    if len(idx) != 1:
        raise ValueError(f"rho={rho} not found uniquely.")
    return int(idx[0])


def _comparison_rows(
    raw: np.ndarray,
    names: list[str],
    noise_levels: np.ndarray,
    rho_values: np.ndarray,
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, float | int | str]] = []

    for sigma_target in SELECTED_SIGMAS:
        noise_idx = int(np.argmin(np.abs(noise_levels - sigma_target)))
        sigma = float(noise_levels[noise_idx])

        for metric in METRICS:
            metric_idx = names.index(metric)
            trial_matrix = raw[noise_idx, :, :, metric_idx]
            means = np.mean(trial_matrix, axis=1)

            best_idx = int(np.argmin(means))
            best_rho = int(rho_values[best_idx])

            comparisons: list[tuple[int, int, str]] = [
                (1, best_rho, "rho1_minus_best"),
                (best_rho, 10, "best_minus_rho10"),
            ]

            if best_rho - 1 >= int(rho_values.min()):
                comparisons.append(
                    (best_rho - 1, best_rho, "left_neighbor_minus_best")
                )
            if best_rho + 1 <= int(rho_values.max()):
                comparisons.append(
                    (best_rho + 1, best_rho, "right_neighbor_minus_best")
                )

            seen: set[tuple[int, int, str]] = set()
            for rho_a, rho_b, label in comparisons:
                key = (rho_a, rho_b, label)
                if key in seen:
                    continue
                seen.add(key)

                ia = _rho_index(rho_values, rho_a)
                ib = _rho_index(rho_values, rho_b)
                x = trial_matrix[ia]
                y = trial_matrix[ib]

                diff_mean, lo, hi, p_better = _paired_bootstrap_ci(
                    x,
                    y,
                    n_bootstrap=N_BOOTSTRAP,
                    rng=rng,
                )

                rows.append(
                    {
                        "sigma": sigma,
                        "metric": metric,
                        "best_rho_by_mean": best_rho,
                        "comparison": label,
                        "rho_a": rho_a,
                        "rho_b": rho_b,
                        "mean_rho_a": float(means[ia]),
                        "mean_rho_b": float(means[ib]),
                        "diff_mean_rho_a_minus_rho_b": diff_mean,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "paired_trial_prob_rho_b_better": p_better,
                        "ci_excludes_zero": bool((lo > 0.0) or (hi < 0.0)),
                    }
                )

    return pd.DataFrame(rows)


def _tail_rows(
    raw: np.ndarray,
    names: list[str],
    noise_levels: np.ndarray,
    rho_values: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for sigma_target in SELECTED_SIGMAS:
        noise_idx = int(np.argmin(np.abs(noise_levels - sigma_target)))
        sigma = float(noise_levels[noise_idx])

        for metric in METRICS:
            metric_idx = names.index(metric)
            trial_matrix = raw[noise_idx, :, :, metric_idx]
            means = np.mean(trial_matrix, axis=1)
            best_idx = int(np.argmin(means))

            selected_indices = sorted(
                set(
                    [
                        _rho_index(rho_values, 1),
                        best_idx,
                        _rho_index(rho_values, 10),
                    ]
                )
            )
            for rho_idx in selected_indices:
                values = trial_matrix[rho_idx]
                rows.append(
                    {
                        "sigma": sigma,
                        "metric": metric,
                        "rho": int(rho_values[rho_idx]),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "q95": float(np.quantile(values, 0.95)),
                        "q99": float(np.quantile(values, 0.99)),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    raw, names, noise_levels, rho_values = _load_real_audit()

    comparisons = _comparison_rows(raw, names, noise_levels, rho_values)
    tails = _tail_rows(raw, names, noise_levels, rho_values)

    comp_path = OUTPUT_DIR / "paired_bootstrap_comparisons.csv"
    tails_path = OUTPUT_DIR / "paired_tail_summary.csv"
    comparisons.to_csv(comp_path, index=False)
    tails.to_csv(tails_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 6)

    print("\nPAIRED BOOTSTRAP COMPARISONS")
    print(comparisons.to_string(index=False))

    print("\nTAIL SUMMARY")
    print(tails.to_string(index=False))

    print()
    print(f"Bootstrap replicates: {N_BOOTSTRAP}")
    print(f"Bootstrap seed:       {BOOTSTRAP_SEED}")
    print(f"Saved comparisons:    {comp_path}")
    print(f"Saved tail summary:   {tails_path}")


if __name__ == "__main__":
    main()
