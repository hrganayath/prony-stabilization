"""Summarize wrapped-metric and circular-noise audit outputs.

Run from repository root after the full audit runs:

    python -m experiments.paired_oversampling.summarize_audit
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import NOISE_LEVELS, RHO_VALUES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "paired_oversampling"


def _load(noise_model: str):
    audit = np.load(OUTPUT_DIR / f"audit_{noise_model}.npz", allow_pickle=False)
    return (
        audit["raw"],
        [str(x) for x in audit["metric_names"]],
        audit["noise_levels"],
        audit["rho_values"],
    )


def _metric(raw: np.ndarray, names: list[str], name: str) -> np.ndarray:
    return raw[..., names.index(name)]


def _row(noise_model: str, raw: np.ndarray, names: list[str], sigma_idx: int, rho_idx: int):
    h = _metric(raw, names, "historical_exponent_error")[sigma_idx, rho_idx]
    w = _metric(raw, names, "wrapped_exponent_error")[sigma_idx, rho_idx]
    ah = _metric(raw, names, "historical_amplitude_error")[sigma_idx, rho_idx]
    aw = _metric(raw, names, "wrapped_match_amplitude_error")[sigma_idx, rho_idx]
    jumps = _metric(raw, names, "branch_jump_count")[sigma_idx, rho_idx]
    disagree = _metric(raw, names, "matching_disagreement")[sigma_idx, rho_idx]

    return {
        "noise_model": noise_model,
        "sigma": float(NOISE_LEVELS[sigma_idx]),
        "rho": int(RHO_VALUES[rho_idx]),
        "hist_exp_mean": float(np.mean(h)),
        "wrap_exp_mean": float(np.mean(w)),
        "wrap_exp_median": float(np.median(w)),
        "wrap_exp_q95": float(np.quantile(w, 0.95)),
        "hist_amp_mean": float(np.mean(ah)),
        "wrapmatch_amp_mean": float(np.mean(aw)),
        "branch_jump_trial_rate": float(np.mean(jumps > 0)),
        "matching_disagreement_rate": float(np.mean(disagree > 0)),
    }


def _best_by_sigma(noise_model: str, raw: np.ndarray, names: list[str]) -> pd.DataFrame:
    wrapped = _metric(raw, names, "wrapped_exponent_error")
    amp = _metric(raw, names, "wrapped_match_amplitude_error")

    rows = []
    for i, sigma in enumerate(NOISE_LEVELS):
        mean_w = np.mean(wrapped[i], axis=1)
        mean_a = np.mean(amp[i], axis=1)
        j_w = int(np.argmin(mean_w))
        j_a = int(np.argmin(mean_a))
        rows.append(
            {
                "noise_model": noise_model,
                "sigma": float(sigma),
                "best_rho_wrapped_exponent": int(RHO_VALUES[j_w]),
                "min_wrapped_exponent_mean": float(mean_w[j_w]),
                "gain_vs_rho1_pct": float(
                    100.0 * (mean_w[0] - mean_w[j_w]) / mean_w[0]
                ) if mean_w[0] > 0 else 0.0,
                "best_rho_amplitude": int(RHO_VALUES[j_a]),
                "min_amplitude_mean": float(mean_a[j_a]),
                "amp_gain_vs_rho1_pct": float(
                    100.0 * (mean_a[0] - mean_a[j_a]) / mean_a[0]
                ) if mean_a[0] > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    all_rows = []
    best_frames = []

    selected_sigmas = [2.0 / 9.0, 0.5]
    selected_rhos = [1, 3, 5, 10]

    for model in ("real", "circular"):
        raw, names, noise_levels, rho_values = _load(model)

        assert np.allclose(noise_levels, NOISE_LEVELS)
        assert np.array_equal(rho_values, RHO_VALUES)

        for sigma in selected_sigmas:
            sigma_idx = int(np.argmin(np.abs(NOISE_LEVELS - sigma)))
            for rho in selected_rhos:
                rho_idx = int(np.where(RHO_VALUES == rho)[0][0])
                all_rows.append(_row(model, raw, names, sigma_idx, rho_idx))

        best_frames.append(_best_by_sigma(model, raw, names))

    selected = pd.DataFrame(all_rows)
    best = pd.concat(best_frames, ignore_index=True)

    selected_path = OUTPUT_DIR / "audit_selected_summary.csv"
    best_path = OUTPUT_DIR / "audit_grid_minima.csv"
    selected.to_csv(selected_path, index=False)
    best.to_csv(best_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.precision", 6)

    print("\nSELECTED CELLS")
    print(selected.to_string(index=False))

    print("\nGRID MINIMA")
    print(best.to_string(index=False))

    # Compact headline comparison at sigma=0.5.
    print("\nHEADLINE sigma=0.5")
    for model in ("real", "circular"):
        row = best[(best["noise_model"] == model) & np.isclose(best["sigma"], 0.5)].iloc[0]
        print(
            f"{model:8s}: wrapped exponent best rho={int(row['best_rho_wrapped_exponent'])}, "
            f"gain={row['gain_vs_rho1_pct']:.2f}%; "
            f"amplitude best rho={int(row['best_rho_amplitude'])}, "
            f"gain={row['amp_gain_vs_rho1_pct']:.2f}%"
        )

    print(f"\nSaved: {selected_path}")
    print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()
