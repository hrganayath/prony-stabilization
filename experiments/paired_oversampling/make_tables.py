"""Generate paper-facing tables from canonical paired-oversampling outputs.

Run after all full canonical studies:

    python -m experiments.paired_oversampling.make_tables

The generator clears old CSVs first so stale paper tables cannot survive a
regeneration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "paired_oversampling"
TABLE_DIR = RESULTS_DIR / "tables"

BASELINE_CSV = RESULTS_DIR / "summary.csv"
AUDIT_REAL_CSV = RESULTS_DIR / "audit_real_summary.csv"
AUDIT_CIRCULAR_CSV = RESULTS_DIR / "audit_circular_summary.csv"
DIAGNOSTICS_CSV = RESULTS_DIR / "diagnostics_summary.csv"
SPECTRAL_CSV = RESULTS_DIR / "spectral_diagnostics_summary.csv"
SCOPE_CSV = RESULTS_DIR / "scope_checks_summary.csv"
DECAY_BLOCKS_CSV = RESULTS_DIR / "signal_decay_blocks.csv"


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found: {path}\n"
            "Run the corresponding full canonical experiment first."
        )


def _mean_table(
    df: pd.DataFrame,
    name_column: str,
    name: str,
) -> pd.DataFrame:
    x = df[df[name_column] == name]
    return x.pivot(index="noise_level", columns="rho", values="mean")


def _mean_value(
    df: pd.DataFrame,
    sigma: float,
    rho: int,
    name_column: str,
    name: str,
) -> float:
    x = df[
        np.isclose(df["noise_level"], sigma)
        & (df["rho"] == rho)
        & (df[name_column] == name)
    ]
    return float(x["mean"].iloc[0])


def build_parameter_optima(audit_real: pd.DataFrame) -> pd.DataFrame:
    """Descriptive grid minima for the primary wrap-aware parameter metrics."""
    rows = []
    metrics = (
        "wrapped_exponent_error",
        "wrapped_match_amplitude_error",
    )

    for sigma in sorted(audit_real["noise_level"].unique()):
        if np.isclose(sigma, 0.0):
            continue

        row: dict[str, float | int] = {"noise_level": float(sigma)}

        for metric in metrics:
            x = audit_real[
                np.isclose(audit_real["noise_level"], sigma)
                & (audit_real["metric"] == metric)
            ].sort_values("rho")

            best = x.loc[x["mean"].idxmin()]
            rho1 = float(x.loc[x["rho"] == 1, "mean"].iloc[0])
            best_mean = float(best["mean"])
            gain_pct = 100.0 * (rho1 - best_mean) / rho1

            prefix = "exponent" if metric == "wrapped_exponent_error" else "amplitude"
            row[f"{prefix}_best_rho"] = int(best["rho"])
            row[f"{prefix}_rho1_mean"] = rho1
            row[f"{prefix}_best_mean"] = best_mean
            row[f"{prefix}_gain_vs_rho1_pct"] = gain_pct

        rows.append(row)

    return pd.DataFrame(rows)


def build_reconstruction_optima(baseline: pd.DataFrame) -> pd.DataFrame:
    """Descriptive minima for fixed-window reconstruction metrics."""
    rows = []

    for sigma in sorted(baseline["noise_level"].unique()):
        if np.isclose(sigma, 0.0):
            continue

        row: dict[str, float | int] = {"noise_level": float(sigma)}

        for metric in ("rmse_clean", "rmse_noisy"):
            x = baseline[
                np.isclose(baseline["noise_level"], sigma)
                & (baseline["metric"] == metric)
            ].sort_values("rho")
            best = x.loc[x["mean"].idxmin()]
            prefix = "clean_rmse" if metric == "rmse_clean" else "noisy_rmse"
            row[f"{prefix}_best_rho"] = int(best["rho"])
            row[f"{prefix}_best_mean"] = float(best["mean"])

        rows.append(row)

    return pd.DataFrame(rows)


def build_selected_noise_table(
    baseline: pd.DataFrame,
    audit_real: pd.DataFrame,
    diagnostics: pd.DataFrame,
    spectral: pd.DataFrame,
) -> pd.DataFrame:
    """Detailed canonical table for sigma=2/9 and sigma=0.5."""
    rows = []

    for sigma in (2.0 / 9.0, 0.5):
        for rho in range(1, 11):
            rows.append(
                {
                    "noise_level": sigma,
                    "rho": rho,
                    "wrapped_exponent_error": _mean_value(audit_real, sigma, rho, "metric", "wrapped_exponent_error"),
                    "wrapped_amplitude_error": _mean_value(audit_real, sigma, rho, "metric", "wrapped_match_amplitude_error"),
                    "rmse_clean": _mean_value(baseline, sigma, rho, "metric", "rmse_clean"),
                    "rmse_noisy": _mean_value(baseline, sigma, rho, "metric", "rmse_noisy"),
                    "hankel_condition": _mean_value(baseline, sigma, rho, "metric", "hankel_condition"),
                    "damping_rmse": _mean_value(diagnostics, sigma, rho, "diagnostic", "damping_rmse"),
                    "frequency_rmse": _mean_value(diagnostics, sigma, rho, "diagnostic", "frequency_rmse"),
                    "oracle_amplitude_error": _mean_value(diagnostics, sigma, rho, "diagnostic", "oracle_amplitude_error"),
                    "right_subspace_angle_max_deg": _mean_value(spectral, sigma, rho, "diagnostic", "right_subspace_angle_max_deg"),
                    "noisy_gap_ratio": _mean_value(spectral, sigma, rho, "diagnostic", "noisy_gap_ratio_sigma_n_over_sigma_np1"),
                    "noise_over_clean_sigma_n": _mean_value(spectral, sigma, rho, "diagnostic", "noise_over_clean_sigma_n"),
                }
            )

    return pd.DataFrame(rows)


def build_noise_model_minima(
    audit_real: pd.DataFrame,
    audit_circular: pd.DataFrame,
) -> pd.DataFrame:
    """Compare descriptive parameter minima under real and circular noise."""
    rows = []

    for label, df in (("real", audit_real), ("circular", audit_circular)):
        for sigma in sorted(df["noise_level"].unique()):
            if np.isclose(sigma, 0.0):
                continue

            for metric in ("wrapped_exponent_error", "wrapped_match_amplitude_error"):
                x = df[
                    np.isclose(df["noise_level"], sigma)
                    & (df["metric"] == metric)
                ].sort_values("rho")
                best = x.loc[x["mean"].idxmin()]
                rho1 = float(x.loc[x["rho"] == 1, "mean"].iloc[0])
                best_mean = float(best["mean"])

                rows.append(
                    {
                        "noise_model": label,
                        "noise_level": float(sigma),
                        "metric": metric,
                        "best_rho": int(best["rho"]),
                        "rho1_mean": rho1,
                        "best_mean": best_mean,
                        "gain_vs_rho1_pct": 100.0 * (rho1 - best_mean) / rho1,
                    }
                )

    return pd.DataFrame(rows)


def build_scope_minima(scope: pd.DataFrame) -> pd.DataFrame:
    """Descriptive minima for the n=6 and n=8 scope checks."""
    rows = []

    for n in sorted(scope["n"].unique()):
        for sigma in sorted(scope["noise_level"].unique()):
            for metric in sorted(scope["metric"].unique()):
                x = scope[
                    (scope["n"] == n)
                    & np.isclose(scope["noise_level"], sigma)
                    & (scope["metric"] == metric)
                ].sort_values("rho")

                best = x.loc[x["mean"].idxmin()]
                rho1 = float(x.loc[x["rho"] == 1, "mean"].iloc[0])
                best_mean = float(best["mean"])

                rows.append(
                    {
                        "n": int(n),
                        "noise_level": float(sigma),
                        "metric": metric,
                        "best_rho": int(best["rho"]),
                        "rho1_mean": rho1,
                        "best_mean": best_mean,
                        "gain_vs_rho1_pct": 100.0 * (rho1 - best_mean) / rho1,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    required = (
        BASELINE_CSV,
        AUDIT_REAL_CSV,
        AUDIT_CIRCULAR_CSV,
        DIAGNOSTICS_CSV,
        SPECTRAL_CSV,
        SCOPE_CSV,
        DECAY_BLOCKS_CSV,
    )
    for path in required:
        _require(path)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for old_csv in TABLE_DIR.glob("*.csv"):
        old_csv.unlink()

    baseline = pd.read_csv(BASELINE_CSV)
    audit_real = pd.read_csv(AUDIT_REAL_CSV)
    audit_circular = pd.read_csv(AUDIT_CIRCULAR_CSV)
    diagnostics = pd.read_csv(DIAGNOSTICS_CSV)
    spectral = pd.read_csv(SPECTRAL_CSV)
    scope = pd.read_csv(SCOPE_CSV)
    decay = pd.read_csv(DECAY_BLOCKS_CSV)

    build_parameter_optima(audit_real).to_csv(TABLE_DIR / "parameter_grid_minima.csv", index=False)
    build_reconstruction_optima(baseline).to_csv(TABLE_DIR / "reconstruction_grid_minima.csv", index=False)
    build_selected_noise_table(baseline, audit_real, diagnostics, spectral).to_csv(
        TABLE_DIR / "selected_noise_diagnostics.csv", index=False
    )
    build_noise_model_minima(audit_real, audit_circular).to_csv(
        TABLE_DIR / "noise_model_minima.csv", index=False
    )
    build_scope_minima(scope).to_csv(TABLE_DIR / "scope_checks_minima.csv", index=False)
    decay.to_csv(TABLE_DIR / "added_sample_block_decay.csv", index=False)

    for metric in ("wrapped_exponent_error", "wrapped_match_amplitude_error"):
        _mean_table(audit_real, "metric", metric).to_csv(TABLE_DIR / f"mean_{metric}.csv")

    for metric in ("rmse_clean", "rmse_noisy", "hankel_condition"):
        _mean_table(baseline, "metric", metric).to_csv(TABLE_DIR / f"mean_{metric}.csv")

    for diagnostic in ("damping_rmse", "frequency_rmse", "oracle_amplitude_error"):
        _mean_table(diagnostics, "diagnostic", diagnostic).to_csv(
            TABLE_DIR / f"mean_{diagnostic}.csv"
        )

    for diagnostic in (
        "right_subspace_angle_max_deg",
        "right_subspace_angle_mean_deg",
        "clean_sigma_n",
        "noisy_gap_ratio_sigma_n_over_sigma_np1",
        "noise_operator_norm",
        "noise_over_clean_sigma_n",
    ):
        _mean_table(spectral, "diagnostic", diagnostic).to_csv(
            TABLE_DIR / f"mean_{diagnostic}.csv"
        )

    print(f"Saved canonical paper-facing tables to: {TABLE_DIR}")


if __name__ == "__main__":
    main()
