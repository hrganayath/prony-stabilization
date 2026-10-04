"""Deterministic signal-decay diagnostic for the paired oversampling study.

This script does not rerun Monte Carlo simulations. It visualizes how the
clean damped signal and its component envelopes evolve over the full sample
horizon used by rho=1,...,10, and it summarizes the energy in each block of
new samples added when rho increases.

Run from the repository root with:

    python -m experiments.paired_oversampling.signal_decay

Outputs:
    results/paired_oversampling/signal_decay.csv
    results/paired_oversampling/signal_decay_blocks.csv
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
BLOCK_CSV_PATH = OUTPUT_DIR / "signal_decay_blocks.csv"
FIGURE_PATH = OUTPUT_DIR / "signal_decay.png"

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
    """Build a deterministic sample-wise table describing signal decay."""
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

    for sigma in REFERENCE_SIGMAS:
        label = str(sigma).replace(".", "p")
        data[f"abs_signal_over_sigma_{label}"] = np.abs(y) / sigma
        data[f"rms_envelope_over_sigma_{label}"] = (
            data["rms_component_envelope"] / sigma
        )

    rho_endpoint = np.full(MAX_SAMPLES, np.nan)
    for rho in RHO_VALUES:
        endpoint = sample_count(int(rho)) - 1
        rho_endpoint[endpoint] = int(rho)
    data["rho_endpoint"] = rho_endpoint

    return pd.DataFrame(data)


def build_added_block_table() -> pd.DataFrame:
    """Summarize the samples newly introduced by each increase in rho.

    rho=1 uses the initial 2n+1 samples. For rho>1, each step adds exactly n
    new samples. Block RMS is more stable than a single endpoint magnitude,
    because the complex signal can oscillate through constructive and
    destructive interference.
    """
    y = clean_signal(MAX_SAMPLES)
    envelopes = component_envelopes(MAX_SAMPLES)

    rows: list[dict[str, float | int]] = []

    for rho in RHO_VALUES:
        rho = int(rho)
        stop = sample_count(rho)

        if rho == 1:
            start = 0
        else:
            start = sample_count(rho - 1)

        signal_block = y[start:stop]
        envelope_block = envelopes[:, start:stop]

        signal_rms = float(
            np.sqrt(np.mean(np.abs(signal_block) ** 2))
        )
        component_rms = float(
            np.sqrt(np.mean(envelope_block**2))
        )

        row: dict[str, float | int] = {
            "rho": rho,
            "k_start": int(start),
            "k_end": int(stop - 1),
            "block_length": int(stop - start),
            "signal_rms": signal_rms,
            "component_envelope_rms": component_rms,
        }

        for sigma in REFERENCE_SIGMAS:
            label = str(sigma).replace(".", "p")
            row[f"signal_rms_over_sigma_{label}"] = signal_rms / sigma
            row[f"component_rms_over_sigma_{label}"] = component_rms / sigma

        rows.append(row)

    return pd.DataFrame(rows)


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


def print_selected_endpoints(df: pd.DataFrame) -> None:
    """Print endpoint values while warning that |s_k| is oscillatory."""
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

    print()
    print(
        "Note: |s_k| is oscillatory because the four complex components "
        "interfere. The added-block RMS table below is the safer decay summary."
    )


def print_added_blocks(blocks: pd.DataFrame) -> None:
    """Print the RMS magnitude of the new data contributed by each rho."""
    print()
    print("RMS magnitude in the newly added sample block")
    print("-" * 86)
    print(
        f"{'rho':>4} {'new k':>10} {'signal RMS':>12} "
        f"{'comp. RMS':>12} {'signal/(2/9)':>14} {'signal/0.5':>12}"
    )

    for _, row in blocks.iterrows():
        interval = f"{int(row['k_start'])}-{int(row['k_end'])}"
        print(
            f"{int(row['rho']):4d} {interval:>10} "
            f"{row['signal_rms']:12.6f} "
            f"{row['component_envelope_rms']:12.6f} "
            f"{row['signal_rms_over_sigma_0p2222222222222222']:14.6f} "
            f"{row['signal_rms_over_sigma_0p5']:12.6f}"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_decay_table()
    blocks = build_added_block_table()

    df.to_csv(CSV_PATH, index=False)
    blocks.to_csv(BLOCK_CSV_PATH, index=False)
    make_figure(df)

    print_selected_endpoints(df)
    print_added_blocks(blocks)

    print()
    print(f"Saved decay table to:       {CSV_PATH}")
    print(f"Saved block summary to:     {BLOCK_CSV_PATH}")
    print(f"Saved decay figure to:      {FIGURE_PATH}")


if __name__ == "__main__":
    main()
