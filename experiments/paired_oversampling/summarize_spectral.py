"""Summarize the comparable spectral diagnostics.

Run after spectral_diagnostics.py:

    python -m experiments.paired_oversampling.summarize_spectral
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import NOISE_LEVELS, RHO_VALUES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "paired_oversampling"


def main() -> None:
    data = np.load(
        OUTPUT_DIR / "spectral_diagnostics.npz",
        allow_pickle=False,
    )
    raw = data["raw"]
    names = [str(x) for x in data["diagnostic_names"]]
    noise_levels = data["noise_levels"]
    rho_values = data["rho_values"]

    assert np.allclose(noise_levels, NOISE_LEVELS)
    assert np.array_equal(rho_values, RHO_VALUES)

    selected_sigmas = [2.0 / 9.0, 0.5]
    selected_rhos = [1, 3, 5, 10]

    rows = []
    for sigma in selected_sigmas:
        i = int(np.argmin(np.abs(NOISE_LEVELS - sigma)))
        for rho in selected_rhos:
            j = int(np.where(RHO_VALUES == rho)[0][0])

            row = {
                "sigma": float(NOISE_LEVELS[i]),
                "rho": int(rho),
            }
            for k, name in enumerate(names):
                values = raw[i, j, :, k]
                row[f"{name}_mean"] = float(np.mean(values))
                row[f"{name}_median"] = float(np.median(values))
            rows.append(row)

    selected = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "spectral_selected_summary.csv"
    selected.to_csv(out_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 6)

    print("\nSELECTED SPECTRAL DIAGNOSTICS")
    print(selected.to_string(index=False))

    print("\nRIGHT-SUBSPACE ANGLE MINIMA")
    angle_idx = names.index("right_subspace_angle_max_deg")
    for sigma in selected_sigmas:
        i = int(np.argmin(np.abs(NOISE_LEVELS - sigma)))
        means = np.mean(raw[i, :, :, angle_idx], axis=1)
        j = int(np.argmin(means))
        print(
            f"sigma={NOISE_LEVELS[i]:.6f}: "
            f"best rho={int(RHO_VALUES[j])}, "
            f"mean max angle={means[j]:.6f} deg"
        )

    print("\nGAP-RATIO TREND")
    gap_idx = names.index("noisy_gap_ratio_sigma_n_over_sigma_np1")
    for sigma in selected_sigmas:
        i = int(np.argmin(np.abs(NOISE_LEVELS - sigma)))
        means = np.mean(raw[i, :, :, gap_idx], axis=1)
        vals = ", ".join(
            f"rho={int(r)}:{v:.3f}"
            for r, v in zip(RHO_VALUES, means)
        )
        print(f"sigma={NOISE_LEVELS[i]:.6f}: {vals}")

    print("\nNOISE/LEAN-SIGNAL MARGIN TREND")
    ratio_idx = names.index("noise_over_clean_sigma_n")
    for sigma in selected_sigmas:
        i = int(np.argmin(np.abs(NOISE_LEVELS - sigma)))
        means = np.mean(raw[i, :, :, ratio_idx], axis=1)
        vals = ", ".join(
            f"rho={int(r)}:{v:.3f}"
            for r, v in zip(RHO_VALUES, means)
        )
        print(f"sigma={NOISE_LEVELS[i]:.6f}: {vals}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
