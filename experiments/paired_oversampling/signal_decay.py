"""Deterministic signal-decay diagnostic for the paired oversampling study.

This script does not rerun Monte Carlo simulations. It visualizes how the
clean damped signal and its component envelopes evolve over the full sample
horizon used by rho=1,...,10.

Run from the repository root with:

    python -m experiments.paired_oversampling.signal_decay

Outputs:
    results/paired_oversampling/signal_decay.csv
    results/paired_oversampling/signal_decay.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    AMPLITUDES,
    EXPONENTS,
    MAX_SAMPLES,
    MODEL_ORDER,
    RHO_VALUES,
    clean_signal,
    sample_count,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "paired_oversampling"
CSV_PATH = OUTPUT_DIR / "signal_decay.csv"
FIGURE_PATH = OUTPUT_DIR / "signal_decay.png"

# Two representative nonzero noise levels already used in the paper analysis.
REFERENCE_SIGMAS = (
    2.0 / 9.0,
    0.5,
)


def component_envelopes(length: int = MAX_SAMPLES) -> np.ndarray:
    """Return |A_j exp(lambda_j k)| for every component and sample index."""
    k = np.arange(length, dtype=float)
    components = (
        AMPLITUDES[:, None]
        * np.exp(EXPONENTS[:, None] * k[None, :])
    )
    return np.abs(components)


def build_decay_table() -> pd.DataFrame:
    """Build a deterministic table describing clean-signal decay."""
    k = np.arange(MAX_SAMPLES, dtype=int)
    y = clean_signal(MAX_SAMPLES)
    envelopes = component_envelopes(MAX_SAMPLES)

    data: dict[str, np.ndarray] = {
        "k": k,
        "abs_signal": np.abs(y),
        "rms_component_envelope": np.sqrt(
            np.mean(envelopes**2, axis=0)
        ),
        "max_component_envelope": np.max(envelopes, axis=0),
        "min_component_envelope": np.min(envelopes, axis=0),
    }

    for j in range(MODEL_ORDER):
        data[f"component_{j + 1}_envelope"] = envelopes[j]

    # This is deliberately called an amplitude ratio, not an SNR, because the
    # baseline experiment adds real Gaussian noise to a complex-valued signal.
    for sigma in REFERENCE_SIGMAS:
        label = str(sigma).replace(".", "p")
        data[f"abs_signal_over_sigma_{label}"] = np.abs(y) / sigma
        data[f"rms_envelope_over_sigma_{label}"] = (
            data["rms_component_envelope"] / sigma
        )

    # Mark the last sample used by each rho.
    rho_endpoint = np.full(MAX_SAMPLES, np.nan)
    for rho in RHO_VALUES:
        endpoint = sample_count(int(rho)) - 1
        rho_endpoint[endpoint] = int(rho)
    data["rho_endpoint"] = rho_endpoint

    return pd.DataFrame(data)


def make_figure(df: pd.DataFrame) -> None:
    """Plot clean-signal magnitude and component envelopes over sample index."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        df["k"],
        df["abs_signal"],
        linewidth=2.0,
        label=r"$|s_k|$",
    )

    for j in range(MODEL_ORDER):
        ax.plot(
            df["k"],
            df[f"component_{j + 1}_envelope"],
            linewidth=1.2,
            alpha=0.8,
            label=fr"$|A_{j + 1}e^{{\lambda_{j + 1}k}}|$",
        )

    for sigma in REFERENCE_SIGMAS:
        ax.axhline(
            sigma,
            linestyle="--",
            linewidth=1.0,
            label=fr"$\sigma={sigma:.3f}$",
        )

    for rho in RHO_VALUES:
        endpoint = sample_count(int(rho)) - 1
        ax.axvline(
            endpoint,
            linewidth=0.5,
            alpha=0.18,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Sample index $k$")
    ax.set_ylabel("Magnitude")
    ax.set_title("Clean-signal decay across the oversampling horizon")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def print_selected_samples(df: pd.DataFrame) -> None:
    """Print compact values at the endpoints associated with selected rho."""
    selected_rho = (1, 3, 5, 10)

    print()
    print("Selected oversampling endpoints")
    print("-" * 72)
    print(
        f"{'rho':>4} {'k_end':>6} {'|s_k|':>12} "
        f"{'RMS env.':>12} {'|s_k|/(2/9)':>14} {'|s_k|/0.5':>12}"
    )

    for rho in selected_rho:
        k_end = sample_count(rho) - 1
        row = df.iloc[k_end]
        print(
            f"{rho:4d} {k_end:6d} "
            f"{row['abs_signal']:12.6f} "
            f"{row['rms_component_envelope']:12.6f} "
            f"{row['abs_signal_over_sigma_0p2222222222222222']:14.6f} "
            f"{row['abs_signal_over_sigma_0p5']:12.6f}"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_decay_table()
    df.to_csv(CSV_PATH, index=False)
    make_figure(df)
    print_selected_samples(df)

    print()
    print(f"Saved decay table to:  {CSV_PATH}")
    print(f"Saved decay figure to: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
