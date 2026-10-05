"""Generate paper-facing figures from canonical paired-oversampling outputs.

Run only after the full canonical experiments have been generated:

    python -m experiments.paired_oversampling.run
    python -m experiments.paired_oversampling.audit_metrics --noise-model real
    python -m experiments.paired_oversampling.audit_metrics --noise-model circular
    python -m experiments.paired_oversampling.diagnostics
    python -m experiments.paired_oversampling.spectral_diagnostics
    python -m experiments.paired_oversampling.scope_checks
    python -m experiments.paired_oversampling.signal_decay
    python -m experiments.paired_oversampling.make_figures

Outputs are written to:
    results/paired_oversampling/figures/

The generator clears old PNGs first so stale paper figures cannot survive a
regeneration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "paired_oversampling"
FIGURE_DIR = RESULTS_DIR / "figures"

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


def _series(
    df: pd.DataFrame,
    sigma: float,
    name_column: str,
    name: str,
) -> pd.DataFrame:
    return df[
        np.isclose(df["noise_level"], sigma)
        & (df[name_column] == name)
    ].sort_values("rho")


def plot_parameter_errors(audit_real: pd.DataFrame) -> None:
    specs = (
        (
            "wrapped_exponent_error",
            "Mean wrap-aware relative exponent error",
            "exponent_error_vs_rho.png",
        ),
        (
            "wrapped_match_amplitude_error",
            "Mean relative amplitude error",
            "amplitude_error_vs_rho.png",
        ),
    )

    for metric, ylabel, filename in specs:
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for sigma in (2.0 / 9.0, 0.5):
            x = _series(audit_real, sigma, "metric", metric)
            ax.plot(x["rho"], x["mean"], marker="o", label=fr"$\sigma={sigma:.3f}$")
        ax.set_xlabel(r"Oversampling factor $\rho$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / filename, dpi=220)
        plt.close(fig)


def plot_reconstruction_tradeoff(baseline: pd.DataFrame) -> None:
    for sigma in (2.0 / 9.0, 0.5):
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for metric, label in (
            ("rmse_clean", "Clean-target RMSE"),
            ("rmse_noisy", "Noisy-target RMSE"),
        ):
            x = _series(baseline, sigma, "metric", metric)
            ax.plot(x["rho"], x["mean"], marker="o", label=label)
        ax.set_xlabel(r"Oversampling factor $\rho$")
        ax.set_ylabel("Mean RMSE on first 9 samples")
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.25)
        ax.legend()
        ax.set_title(fr"$\sigma={sigma:.3f}$")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"reconstruction_tradeoff_sigma_{sigma:.3f}.png", dpi=220)
        plt.close(fig)


def plot_condition_vs_amplitude(
    baseline: pd.DataFrame,
    audit_real: pd.DataFrame,
) -> None:
    sigma = 0.5
    cond = _series(baseline, sigma, "metric", "hankel_condition")
    amp = _series(audit_real, sigma, "metric", "wrapped_match_amplitude_error")
    fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
    ax2 = ax1.twinx()
    l1 = ax1.plot(cond["rho"], cond["mean"], marker="o", label=r"$\kappa_2(H_\rho)$")
    l2 = ax2.plot(amp["rho"], amp["mean"], marker="s", linestyle="--", label="Amplitude error")
    ax1.set_xlabel(r"Oversampling factor $\rho$")
    ax1.set_ylabel(r"Mean $\kappa_2(H_\rho)$")
    ax2.set_ylabel("Mean relative amplitude error")
    ax1.set_xticks(range(1, 11))
    ax1.grid(True, alpha=0.25)
    lines = l1 + l2
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "condition_vs_amplitude_sigma_0p5.png", dpi=220)
    plt.close(fig)


def plot_component_errors(diagnostics: pd.DataFrame) -> None:
    for sigma in (2.0 / 9.0, 0.5):
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for diagnostic, label in (
            ("damping_rmse", "Damping RMSE"),
            ("frequency_rmse", "Wrapped frequency RMSE"),
        ):
            x = _series(diagnostics, sigma, "diagnostic", diagnostic)
            ax.plot(x["rho"], x["mean"], marker="o", label=label)
        ax.set_xlabel(r"Oversampling factor $\rho$")
        ax.set_ylabel("Mean component error")
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.25)
        ax.legend()
        ax.set_title(fr"$\sigma={sigma:.3f}$")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"component_errors_sigma_{sigma:.3f}.png", dpi=220)
        plt.close(fig)


def plot_oracle_amplitude(
    audit_real: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    sigma = 0.5
    amp = _series(audit_real, sigma, "metric", "wrapped_match_amplitude_error")
    oracle = _series(diagnostics, sigma, "diagnostic", "oracle_amplitude_error")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(amp["rho"], amp["mean"], marker="o", label="Estimated-pole amplitude error")
    ax.plot(oracle["rho"], oracle["mean"], marker="s", linestyle="--", label="Oracle-pole amplitude error")
    ax.set_xlabel(r"Oversampling factor $\rho$")
    ax.set_ylabel("Mean relative amplitude error")
    ax.set_xticks(range(1, 11))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "oracle_amplitude_sigma_0p5.png", dpi=220)
    plt.close(fig)


def plot_right_subspace_angle(spectral: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for sigma in (2.0 / 9.0, 0.5):
        x = _series(spectral, sigma, "diagnostic", "right_subspace_angle_max_deg")
        ax.plot(x["rho"], x["mean"], marker="o", label=fr"$\sigma={sigma:.3f}$")
    ax.set_xlabel(r"Oversampling factor $\rho$")
    ax.set_ylabel("Mean largest right-subspace angle (degrees)")
    ax.set_xticks(range(1, 11))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "right_subspace_angle_vs_rho.png", dpi=220)
    plt.close(fig)


def plot_spectral_balance(spectral: pd.DataFrame) -> None:
    for sigma in (2.0 / 9.0, 0.5):
        gap = _series(spectral, sigma, "diagnostic", "noisy_gap_ratio_sigma_n_over_sigma_np1")
        ratio = _series(spectral, sigma, "diagnostic", "noise_over_clean_sigma_n")
        fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
        ax2 = ax1.twinx()
        l1 = ax1.plot(gap["rho"], gap["mean"], marker="o", label=r"$\sigma_n(H)/\sigma_{n+1}(H)$")
        l2 = ax2.plot(ratio["rho"], ratio["mean"], marker="s", linestyle="--", label=r"$\|E\|_2/\sigma_n(H_{\rm clean})$")
        ax1.set_xlabel(r"Oversampling factor $\rho$")
        ax1.set_ylabel("Mean noisy singular-value separation")
        ax2.set_ylabel("Mean relative Hankel perturbation")
        ax1.set_xticks(range(1, 11))
        ax1.grid(True, alpha=0.25)
        lines = l1 + l2
        ax1.legend(lines, [line.get_label() for line in lines], loc="best")
        ax1.set_title(fr"$\sigma={sigma:.3f}$")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"spectral_balance_sigma_{sigma:.3f}.png", dpi=220)
        plt.close(fig)


def plot_noise_model_comparison(
    audit_real: pd.DataFrame,
    audit_circular: pd.DataFrame,
) -> None:
    sigma = 0.5
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for df, label in (
        (audit_real, "Real Gaussian noise"),
        (audit_circular, "Circular complex Gaussian noise"),
    ):
        x = _series(df, sigma, "metric", "wrapped_exponent_error")
        ax.plot(x["rho"], x["mean"], marker="o", label=label)
    ax.set_xlabel(r"Oversampling factor $\rho$")
    ax.set_ylabel("Mean wrap-aware relative exponent error")
    ax.set_xticks(range(1, 11))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "noise_model_comparison_sigma_0p5.png", dpi=220)
    plt.close(fig)


def plot_scope_checks(scope: pd.DataFrame) -> None:
    for metric, ylabel, filename in (
        ("wrapped_frequency_rmse", "Mean wrapped frequency RMSE (cycles/sample)", "scope_frequency_vs_rho.png"),
        ("relative_amplitude_error", "Mean relative amplitude error", "scope_amplitude_vs_rho.png"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for n in (6, 8):
            for sigma, linestyle in ((2.0 / 9.0, "-"), (0.5, "--")):
                x = scope[
                    (scope["n"] == n)
                    & np.isclose(scope["noise_level"], sigma)
                    & (scope["metric"] == metric)
                ].sort_values("rho")
                ax.plot(
                    x["rho"], x["mean"], marker="o", linestyle=linestyle,
                    label=fr"$n={n},\ \sigma={sigma:.3f}$",
                )
        ax.set_xlabel(r"Oversampling factor $\rho$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / filename, dpi=220)
        plt.close(fig)


def plot_added_block_decay(decay: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(decay["rho"], decay["signal_rms"], marker="o", label="RMS of newly added signal block")
    ax.plot(decay["rho"], decay["component_envelope_rms"], marker="s", linestyle="--", label="RMS component envelope")
    ax.axhline(2.0 / 9.0, linestyle=":", label=r"$\sigma=2/9$")
    ax.axhline(0.5, linestyle="-.", label=r"$\sigma=0.5$")
    ax.set_yscale("log")
    ax.set_xlabel(r"Oversampling factor $\rho$")
    ax.set_ylabel("RMS magnitude")
    ax.set_xticks(range(1, 11))
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "added_block_decay.png", dpi=220)
    plt.close(fig)


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

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for old_png in FIGURE_DIR.glob("*.png"):
        old_png.unlink()

    baseline = pd.read_csv(BASELINE_CSV)
    audit_real = pd.read_csv(AUDIT_REAL_CSV)
    audit_circular = pd.read_csv(AUDIT_CIRCULAR_CSV)
    diagnostics = pd.read_csv(DIAGNOSTICS_CSV)
    spectral = pd.read_csv(SPECTRAL_CSV)
    scope = pd.read_csv(SCOPE_CSV)
    decay = pd.read_csv(DECAY_BLOCKS_CSV)

    plot_parameter_errors(audit_real)
    plot_reconstruction_tradeoff(baseline)
    plot_condition_vs_amplitude(baseline, audit_real)
    plot_component_errors(diagnostics)
    plot_oracle_amplitude(audit_real, diagnostics)
    plot_right_subspace_angle(spectral)
    plot_spectral_balance(spectral)
    plot_noise_model_comparison(audit_real, audit_circular)
    plot_scope_checks(scope)
    plot_added_block_decay(decay)

    print(f"Saved canonical paper-facing figures to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
