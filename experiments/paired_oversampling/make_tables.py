"""Generate paper-facing tables from verified paired-oversampling outputs.

Run after the full baseline and diagnostic studies:

    python -m experiments.paired_oversampling.make_tables

Inputs:
    results/paired_oversampling/summary.csv
    results/paired_oversampling/diagnostics_summary.csv
    results/paired_oversampling/signal_decay_blocks.csv

Outputs:
    results/paired_oversampling/tables/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "paired_oversampling"
TABLE_DIR = RESULTS_DIR / "tables"

BASELINE_CSV = RESULTS_DIR / "summary.csv"
DIAGNOSTICS_CSV = RESULTS_DIR / "diagnostics_summary.csv"
DECAY_BLOCKS_CSV = RESULTS_DIR / "signal_decay_blocks.csv"


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found: {path}\n"
            "Run the corresponding full experiment first."
        )


def _metric_mean_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    x = df[df["metric"] == metric]
    return x.pivot(index="noise_level", columns="rho", values="mean")


def _diagnostic_mean_table(df: pd.DataFrame, diagnostic: str) -> pd.DataFrame:
    x = df[df["diagnostic"] == diagnostic]
    return x.pivot(index="noise_level", columns="rho", values="mean")


def build_optima_table(baseline: pd.DataFrame) -> pd.DataFrame:
    """Descriptive grid minima for nonzero noise levels.

    These are descriptive minima over rho=1,...,10, not statistically
    established unique optima.
    """
    rows = []

    metrics = (
        "exponent_error",
        "amplitude_error",
        "rmse_clean",
        "rmse_noisy",
    )

    for sigma in sorted(baseline["noise_level"].unique()):
        if np.isclose(sigma, 0.0):
            continue

        row: dict[str, float | int] = {"noise_level": float(sigma)}

        for metric in metrics:
            x = baseline[
                (baseline["noise_level"] == sigma)
                & (baseline["metric"] == metric)
            ].sort_values("rho")

            best_idx = x["mean"].idxmin()
            best = x.loc[best_idx]
            rho_best = int(best["rho"])
            value_best = float(best["mean"])

            rho1 = float(x.loc[x["rho"] == 1, "mean"].iloc[0])
            gain = (rho1 - value_best) / rho1 if rho1 != 0 else np.nan

            row[f"{metric}_rho"] = rho_best
            row[f"{metric}_min"] = value_best
            row[f"{metric}_gain_vs_rho1"] = gain

        rows.append(row)

    return pd.DataFrame(rows)


def build_selected_noise_table(
    baseline: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Compact table for sigma=2/9 and sigma=0.5."""
    selected = [2.0 / 9.0, 0.5]
    rows = []

    for sigma in selected:
        for rho in range(1, 11):
            b = baseline[
                np.isclose(baseline["noise_level"], sigma)
                & (baseline["rho"] == rho)
            ]
            d = diagnostics[
                np.isclose(diagnostics["noise_level"], sigma)
                & (diagnostics["rho"] == rho)
            ]

            def bm(name: str) -> float:
                return float(b.loc[b["metric"] == name, "mean"].iloc[0])

            def dm(name: str) -> float:
                return float(
                    d.loc[d["diagnostic"] == name, "mean"].iloc[0]
                )

            rows.append(
                {
                    "noise_level": sigma,
                    "rho": rho,
                    "exponent_error": bm("exponent_error"),
                    "amplitude_error": bm("amplitude_error"),
                    "rmse_clean": bm("rmse_clean"),
                    "rmse_noisy": bm("rmse_noisy"),
                    "hankel_condition": bm("hankel_condition"),
                    "damping_rmse": dm("damping_rmse"),
                    "frequency_rmse": dm("frequency_rmse"),
                    "oracle_amplitude_error": dm("oracle_amplitude_error"),
                    "subspace_angle_max_deg": dm("subspace_angle_max_deg"),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    for path in (BASELINE_CSV, DIAGNOSTICS_CSV, DECAY_BLOCKS_CSV):
        _require(path)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(BASELINE_CSV)
    diagnostics = pd.read_csv(DIAGNOSTICS_CSV)
    decay = pd.read_csv(DECAY_BLOCKS_CSV)

    optima = build_optima_table(baseline)
    selected = build_selected_noise_table(baseline, diagnostics)

    optima.to_csv(TABLE_DIR / "descriptive_grid_minima.csv", index=False)
    selected.to_csv(TABLE_DIR / "selected_noise_diagnostics.csv", index=False)
    decay.to_csv(TABLE_DIR / "added_sample_block_decay.csv", index=False)

    for metric in (
        "exponent_error",
        "amplitude_error",
        "rmse_clean",
        "rmse_noisy",
        "hankel_condition",
    ):
        _metric_mean_table(baseline, metric).to_csv(
            TABLE_DIR / f"mean_{metric}.csv"
        )

    for diagnostic in (
        "damping_rmse",
        "frequency_rmse",
        "oracle_amplitude_error",
        "subspace_angle_max_deg",
    ):
        _diagnostic_mean_table(diagnostics, diagnostic).to_csv(
            TABLE_DIR / f"mean_{diagnostic}.csv"
        )

    print(f"Saved paper-facing tables to: {TABLE_DIR}")


if __name__ == "__main__":
    main()
