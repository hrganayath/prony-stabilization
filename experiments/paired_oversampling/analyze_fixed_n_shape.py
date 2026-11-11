"""Analyze the complete fixed-N Hankel-shape control.

This script does not rerun Monte Carlo experiments. It reads the existing
fixed_n_shape_summary.csv and produces full C-curves plus compact comparison
tables for the two primary metrics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "results" / "paired_oversampling"

INPUT = RESULT_DIR / "fixed_n_shape_summary.csv"
CURVES_CSV = RESULT_DIR / "fixed_n_shape_full_curves.csv"
BEST_CSV = RESULT_DIR / "fixed_n_shape_best_vs_N.csv"
BALANCED_CSV = RESULT_DIR / "fixed_n_shape_balanced_vs_narrow.csv"
FIG_EXP = RESULT_DIR / "fixed_n_shape_exponent_curves.png"
FIG_AMP = RESULT_DIR / "fixed_n_shape_amplitude_curves.png"

METRICS = ("wrapped_exponent_error", "relative_amplitude_error")
N_VALUES = (17, 29, 45)
NOISE_LEVELS = (2.0 / 9.0, 0.5)

def load_summary():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}")
    return pd.read_csv(INPUT)

def select_curves(df):
    mask_metric = df["metric"].isin(METRICS)
    mask_N = df["N"].isin(N_VALUES)
    mask_sigma = df["noise_level"].apply(
        lambda x: any(np.isclose(x, s) for s in NOISE_LEVELS)
    )
    out = df[mask_metric & mask_N & mask_sigma].copy()
    out = out.sort_values(["metric", "noise_level", "N", "columns"])
    out.to_csv(CURVES_CSV, index=False)
    return out

def make_best_vs_n(curves):
    rows = []
    for sigma in NOISE_LEVELS:
        for metric in METRICS:
            for N in N_VALUES:
                b = curves[
                    np.isclose(curves["noise_level"], sigma)
                    & (curves["metric"] == metric)
                    & (curves["N"] == N)
                ].sort_values("columns")
                narrow = b[b["columns"] == 5].iloc[0]
                best = b.loc[b["mean"].idxmin()]
                rows.append({
                    "noise_level": float(sigma),
                    "metric": metric,
                    "N": int(N),
                    "narrow_mean": float(narrow["mean"]),
                    "best_C": int(best["columns"]),
                    "best_rows": int(best["rows"]),
                    "best_mean": float(best["mean"]),
                    "gain_best_vs_narrow_pct": float(
                        100.0 * (narrow["mean"] - best["mean"]) / narrow["mean"]
                    ),
                    "narrow_median": float(narrow["median"]),
                    "best_median": float(best["median"]),
                })
    out = pd.DataFrame(rows)
    out.to_csv(BEST_CSV, index=False)
    return out

def make_balanced(curves):
    rows = []
    for sigma in NOISE_LEVELS:
        for metric in METRICS:
            for N in N_VALUES:
                b = curves[
                    np.isclose(curves["noise_level"], sigma)
                    & (curves["metric"] == metric)
                    & (curves["N"] == N)
                ].sort_values("columns")
                narrow = b[b["columns"] == 5].iloc[0]
                balanced = b.iloc[-1]
                rows.append({
                    "noise_level": float(sigma),
                    "metric": metric,
                    "N": int(N),
                    "narrow_mean": float(narrow["mean"]),
                    "balanced_C": int(balanced["columns"]),
                    "balanced_rows": int(balanced["rows"]),
                    "balanced_mean": float(balanced["mean"]),
                    "balanced_change_vs_narrow_pct": float(
                        100.0 * (balanced["mean"] - narrow["mean"]) / narrow["mean"]
                    ),
                })
    out = pd.DataFrame(rows)
    out.to_csv(BALANCED_CSV, index=False)
    return out

def plot_metric(curves, metric, ylabel, path):
    fig, ax = plt.subplots()
    for sigma in NOISE_LEVELS:
        for N in N_VALUES:
            b = curves[
                np.isclose(curves["noise_level"], sigma)
                & (curves["metric"] == metric)
                & (curves["N"] == N)
            ].sort_values("columns")
            ax.plot(
                b["columns"],
                b["mean"],
                marker="o",
                label=f"N={N}, sigma={sigma:.4f}",
            )
    ax.set_xlabel("Hankel column count C")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def main():
    df = load_summary()
    curves = select_curves(df)
    best = make_best_vs_n(curves)
    balanced = make_balanced(curves)

    plot_metric(
        curves, "wrapped_exponent_error",
        "Mean wrapped exponent error", FIG_EXP
    )
    plot_metric(
        curves, "relative_amplitude_error",
        "Mean relative amplitude error", FIG_AMP
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 6)

    for sigma in NOISE_LEVELS:
        print()
        print("=" * 92)
        print(f"FULL FIXED-N CURVES: sigma={sigma:.6f}")
        print("=" * 92)
        for metric in METRICS:
            print()
            print(metric)
            b = curves[
                np.isclose(curves["noise_level"], sigma)
                & (curves["metric"] == metric)
            ][
                ["N", "columns", "rows", "aspect_rows_over_columns",
                 "mean", "median", "std", "q95"]
            ]
            print(b.to_string(index=False))

    print()
    print("=" * 92)
    print("BEST TESTED WIDTH VERSUS N")
    print("=" * 92)
    print(best.to_string(index=False))

    print()
    print("=" * 92)
    print("NEAR-BALANCED WIDTH VERSUS C=5")
    print("=" * 92)
    print(balanced.to_string(index=False))

    print()
    print(f"Saved full curves:      {CURVES_CSV}")
    print(f"Saved best-vs-N table:  {BEST_CSV}")
    print(f"Saved balanced table:   {BALANCED_CSV}")
    print(f"Saved exponent figure:  {FIG_EXP}")
    print(f"Saved amplitude figure: {FIG_AMP}")

if __name__ == "__main__":
    main()
