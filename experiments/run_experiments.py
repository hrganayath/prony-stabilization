# Run as: python -m experiments.run_experiments

import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")

from prony import (
    compute_residual_error,
    relative_exponent_error,
    match_estimates,
    prony_method,
)
from prony.data import generate_clean_data, generate_noisy_data

from .comparison_tables import save_comparison_tables
from .plotting import (
    interactive_amplitude_error_plot,
    interactive_frequency_error_plot,
    plot_condition_number_by_noise,
    plot_condition_number_heatmap,
    plot_condition_vs_error_scatter,
    plot_error_heatmap,
    plot_error_ratios,
    plot_error_subplots,
    plot_metric_vs_rho_selected_sigmas,
)
from .sensitivity_analysis import plot_sensitivity


# ============================================================================
# Configuration dataclass
# ============================================================================

@dataclass
class ExperimentConfig:
    """All tunable parameters for the Monte Carlo experiment."""
    a_true: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.1, 0.9, 0.7], dtype=complex)
    )
    sigma: np.ndarray = field(
        default_factory=lambda: np.array([-0.10, -0.07, -0.12, -0.08])
    )
    f: np.ndarray = field(
        default_factory=lambda: np.array([0.10, -0.22, 0.33, -0.41])
    )
    noise_levels: np.ndarray = field(
        default_factory=lambda: np.linspace(0.0, 0.5, num=10)
    )
    os_factors: list = field(
        default_factory=lambda: list(range(1, 11))
    )
    num_trials: int = 1000
    global_seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("results"))


# ============================================================================
# Default paths (derived at runtime inside functions, not at import)
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"

RESULTS_CSV         = _DEFAULT_OUTPUT_DIR / "aggregated_results_with_condition.csv"
FREQ_TABLE_XLSX     = _DEFAULT_OUTPUT_DIR / "frequency_error_table.xlsx"
AMP_TABLE_XLSX      = _DEFAULT_OUTPUT_DIR / "amplitude_error_table.xlsx"
COMPARISON_XLSX     = _DEFAULT_OUTPUT_DIR / "comparison_tables.xlsx"
FIGURE_RMSE_HEATMAP = _DEFAULT_OUTPUT_DIR / "figure_rmse_mean_heatmap.png"
FIGURE_RMSE_VS_RHO  = _DEFAULT_OUTPUT_DIR / "figure_rmse_vs_rho_selected_sigmas.png"
FIGURE_COND_HEATMAP = _DEFAULT_OUTPUT_DIR / "figure_conditionnumber_hankel_heatmap.png"
FIGURE_AMP_VS_NOISE = _DEFAULT_OUTPUT_DIR / "figure_amplitude_error_vs_noise.png"
FIGURE_FREQ_VS_NOISE = _DEFAULT_OUTPUT_DIR / "figure_exponent_error_vs_noise.png"


# ============================================================================
# Main experiment runner
# ============================================================================

def run_experiments(config: ExperimentConfig | None = None) -> pd.DataFrame:
    """
    Run Monte Carlo experiments for each oversampling factor and noise level.

    For each configuration (os_factor, noise_level) the following are recorded:
        - Amplitude error (relative)
        - Frequency error (relative, with appropriate handling)
        - Residual / data misfit (full window and fixed window)
        - RMSE (full and fixed)
        - Condition number of the Hankel matrix
        - Error ratios (forward / residual)

    Parameters
    ----------
    config : ExperimentConfig, optional
        Experiment configuration. If None, uses default parameters.

    Returns
    -------
    df_results : pd.DataFrame
        Aggregated results (means and standard deviations) for each
        configuration.
    """
    if config is None:
        config = ExperimentConfig()

    # Resolve output directory and create it now (not at import time)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive signal parameters
    omegas_true = config.sigma + 1j * 2 * np.pi * config.f
    N = len(config.a_true)

    max_required_length = max(config.os_factors) * N + N + 1
    t = np.arange(max_required_length)
    y_clean = generate_clean_data(t, config.a_true, omegas_true)

    # Validate a_true norm before starting
    norm_a_true = np.linalg.norm(config.a_true)
    if norm_a_true < np.finfo(float).eps:
        raise ValueError("a_true has near-zero norm — amplitude error is undefined.")

    # SeedSequence for reproducible, independent per-trial RNG streams
    ss = np.random.SeedSequence(config.global_seed)
    total_configs = len(config.os_factors) * len(config.noise_levels)
    child_seeds = ss.spawn(total_configs * config.num_trials)

    results = []
    total_start = time.perf_counter()
    config_idx = 0
    seed_idx = 0

    for os_factor in config.os_factors:
        for noise_level in config.noise_levels:
            config_idx += 1
            print(
                f"[{config_idx}/{total_configs}] "
                f"os_factor={os_factor}, noise_level={noise_level:.3f}",
                flush=True,
            )

            amp_errors, freq_errors = [], []
            back_errors, rel_back_errors = [], []
            cond_numbers = []
            amp_error_ratios, freq_error_ratios = [], []
            rmse_list, rel_rmse_list = [], []
            back_errors_fixed, rel_back_errors_fixed = [], []
            rmse_fixed_list, rel_rmse_fixed_list = [], []
            runtimes = []

            m = os_factor * N
            L_fixed = 2 * N + 1

            for trial in range(config.num_trials):
                # Each trial gets a unique, independent RNG stream
                rng = np.random.default_rng(child_seeds[seed_idx])
                seed_idx += 1

                y_noisy = generate_noisy_data(
                    y_clean[:m + N + 1],
                    noise_level,
                    rng=rng,
                )

                start_time = time.perf_counter()
                a_est, omegas_est, cond_num = prony_method(
                    y_noisy, os_factor, n=N
                )
                runtimes.append(time.perf_counter() - start_time)

                a_est_m, omegas_est_m, _ = match_estimates(
                    config.a_true, omegas_true, a_est, omegas_est
                )

                amp_err = (
                    np.linalg.norm(config.a_true - a_est_m) / norm_a_true
                )
                amp_errors.append(amp_err)

                freq_err = relative_exponent_error(omegas_true, omegas_est_m)
                freq_errors.append(freq_err)

                cond_numbers.append(cond_num)

                metrics = compute_residual_error(
                    y_noisy, a_est_m, omegas_est_m,
                    oversampling_factor=os_factor, n=N,
                )
                back_errors.append(metrics.residual_norm)
                rel_back_errors.append(metrics.relative_residual)
                rmse_list.append(metrics.rmse)
                rel_rmse_list.append(metrics.rel_rmse)

                metrics_fixed = compute_residual_error(
                    y_noisy, a_est_m, omegas_est_m,
                    oversampling_factor=os_factor, n=N,
                    L_eval=L_fixed,
                )
                back_errors_fixed.append(metrics_fixed.residual_norm)
                rel_back_errors_fixed.append(metrics_fixed.relative_residual)
                rmse_fixed_list.append(metrics_fixed.rmse)
                rel_rmse_fixed_list.append(metrics_fixed.rel_rmse)

                if metrics.residual_norm != 0:
                    amp_error_ratios.append(amp_err / metrics.residual_norm)
                    freq_error_ratios.append(freq_err / metrics.residual_norm)
                else:
                    amp_error_ratios.append(np.nan)
                    freq_error_ratios.append(np.nan)

            results.append({
                "Oversampling Factor": os_factor,
                "Noise Level": noise_level,
                "Amplitude Error Mean": np.mean(amp_errors),
                "Amplitude Error Std": np.std(amp_errors),
                "Frequency Error Mean": np.mean(freq_errors),
                "Frequency Error Std": np.std(freq_errors),
                "Residual Mean": np.mean(back_errors),
                "Residual Std": np.std(back_errors),
                "Relative Residual Mean": np.mean(rel_back_errors),
                "Relative Residual Std": np.std(rel_back_errors),
                "Condition Number Mean": np.mean(cond_numbers),
                "Condition Number Std": np.std(cond_numbers),
                "Condition Number Min": np.min(cond_numbers),
                "Condition Number Max": np.max(cond_numbers),
                "Condition Number Median": np.median(cond_numbers),
                "Amplitude Error Ratio Mean": np.nanmean(amp_error_ratios),
                "Amplitude Error Ratio Std": np.nanstd(amp_error_ratios),
                "Frequency Error Ratio Mean": np.nanmean(freq_error_ratios),
                "Frequency Error Ratio Std": np.nanstd(freq_error_ratios),
                "RMSE Mean": np.mean(rmse_list),
                "RMSE Std": np.std(rmse_list),
                "Rel RMSE Mean": np.mean(rel_rmse_list),
                "Rel RMSE Std": np.std(rel_rmse_list),
                "Residual Fixed Mean": np.mean(back_errors_fixed),
                "Residual Fixed Std": np.std(back_errors_fixed),
                "Relative Residual Fixed Mean": np.mean(rel_back_errors_fixed),
                "Relative Residual Fixed Std": np.std(rel_back_errors_fixed),
                "RMSE Fixed Mean": np.mean(rmse_fixed_list),
                "RMSE Fixed Std": np.std(rmse_fixed_list),
                "Rel RMSE Fixed Mean": np.mean(rel_rmse_fixed_list),
                "Rel RMSE Fixed Std": np.std(rel_rmse_fixed_list),
                "Runtime Mean (s)": float(np.mean(runtimes)),
                "Runtime Std (s)": float(np.std(runtimes)),
            })

    total_time = time.perf_counter() - total_start
    print(f"\nTotal experiment time: {total_time:.2f} seconds")
    print(f"Saving outputs to: {output_dir}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / "aggregated_results_with_condition.csv", index=False)
    return df_results


# ============================================================================
# Plotting
# ============================================================================

def plot_and_save_results(
    df_results: pd.DataFrame,
    config: ExperimentConfig | None = None,
) -> None:
    """Generate all plots and save them (both static and interactive)."""
    if config is None:
        config = ExperimentConfig()

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    noise_levels = config.noise_levels
    os_factors = config.os_factors

    table_frequency = df_results.pivot(
        index="Noise Level",
        columns="Oversampling Factor",
        values="Frequency Error Mean",
    )
    table_amplitude = df_results.pivot(
        index="Noise Level",
        columns="Oversampling Factor",
        values="Amplitude Error Mean",
    )

    print("\nFrequency error table (mean):")
    print(table_frequency.to_markdown())
    table_frequency.to_excel(
        output_dir / "frequency_error_table.xlsx",
        sheet_name="Frequency Errors",
    )

    print("\nAmplitude error table (mean):")
    print(table_amplitude.to_markdown())
    table_amplitude.to_excel(
        output_dir / "amplitude_error_table.xlsx",
        sheet_name="Amplitude Errors",
    )

    # Amplitude error vs noise — save to disk instead of plt.show()
    plt.figure(figsize=(15, 6))
    for osf in os_factors:
        subset = df_results[df_results["Oversampling Factor"] == osf]
        plt.plot(
            subset["Noise Level"],
            subset["Amplitude Error Mean"],
            label=f"ρ = {osf}",
            marker="o",
        )
    plt.title("Amplitude Error vs Noise Level")
    plt.xlabel("Noise Level")
    plt.ylabel("Amplitude Error (Mean)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.savefig(
        output_dir / "figure_amplitude_error_vs_noise.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Exponent error vs noise — save to disk instead of plt.show()
    plt.figure(figsize=(15, 6))
    for osf in os_factors:
        subset = df_results[df_results["Oversampling Factor"] == osf]
        plt.errorbar(
            subset["Noise Level"],
            subset["Frequency Error Mean"],
            yerr=subset["Frequency Error Std"],
            label=f"ρ = {osf}",
            capsize=5,
            marker="o",
        )
    plt.title("Exponent Error vs Noise Level")
    plt.xlabel("Noise Level")
    plt.ylabel("Exponent Error (Mean ± Std)")
    plt.legend()
    plt.savefig(
        output_dir / "figure_exponent_error_vs_noise.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    plot_error_heatmap(
        df_results,
        "Residual Mean",
        "Residual Heatmap (Mean)\n(Noise Level vs Oversampling Factor)",
    )
    plot_error_heatmap(
        df_results,
        "RMSE Fixed Mean",
        "Fixed-window RMSE Heatmap (Mean)\n(Noise Level vs Oversampling Factor)",
    )
    plot_error_heatmap(
        df_results,
        "RMSE Mean",
        "RMSE Heatmap (Mean)\n(Noise Level vs Oversampling Factor)",
        save_path=output_dir / "figure_rmse_mean_heatmap.png",
    )

    df_renamed = df_results.rename(
        columns={
            "Residual Mean": "Residual",
            "Relative Residual Mean": "Relative Residual",
        }
    )
    plot_error_subplots(df_renamed)

    # Derive sigma_targets from actual noise_levels to guarantee they exist
    sigma_targets = [
        noise_levels[2],
        noise_levels[4],
        noise_levels[6],
        noise_levels[-1],
    ]
    plot_metric_vs_rho_selected_sigmas(
        df_results,
        "RMSE Mean",
        sigma_targets=sigma_targets,
        title="RMSE vs Oversampling Factor (selected noise levels)",
        save_path=output_dir / "figure_rmse_vs_rho_selected_sigmas.png",
    )

    grouped = df_results.groupby("Oversampling Factor").mean(numeric_only=True)
    amp_ratios_mean = grouped["Amplitude Error Ratio Mean"].values
    freq_ratios_mean = grouped["Frequency Error Ratio Mean"].values
    plot_error_ratios(os_factors, amp_ratios_mean, freq_ratios_mean)

    plot_condition_number_heatmap(
        df_results,
        title=r"Condition number of Hankel matrix ($\log_{10}\kappa(H)$)",
        save_path=output_dir / "figure_conditionnumber_hankel_heatmap.png",
        drop_sigma0=True,
    )
    plot_condition_vs_error_scatter(df_results)
    plot_condition_number_by_noise(df_results)

    try:
        interactive_amplitude_error_plot(df_results)
        interactive_frequency_error_plot(df_results)
    except Exception as e:
        print("Interactive plotting encountered an error:", e)


# ============================================================================
# Entry point
# ============================================================================

def main(config: ExperimentConfig | None = None) -> None:
    """Run the full experiment suite and produce all outputs."""
    if config is None:
        config = ExperimentConfig()

    omegas_true = config.sigma + 1j * 2 * np.pi * config.f
    t = np.arange(max(config.os_factors) * len(config.a_true) + len(config.a_true) + 1)

    df_results = run_experiments(config)
    plot_and_save_results(df_results, config)
    save_comparison_tables(
        df_results,
        excel_path=config.output_dir / "comparison_tables.xlsx",
    )
    plot_sensitivity(config.a_true, omegas_true, t, config.os_factors)


if __name__ == "__main__":
    main()
