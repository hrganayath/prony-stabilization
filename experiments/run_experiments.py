# Run as: python -m experiments.run_experiments

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")

# Core Prony functions
from prony import (
    compute_backward_error,
    frequency_error_generic,
    match_estimates,
    prony_method,
)
from prony.data import generate_clean_data, generate_noisy_data

# Local modules (plotting, tables, sensitivity)
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
# Experiment configuration
# ============================================================================
# Active true parameters (4-exponential case)
a_true = np.array([1.0, 1.1, 0.9, 0.7], dtype=complex)
sigma = np.array([-0.10, -0.07, -0.12, -0.08])  # decays (Re ω < 0)
f = np.array([0.10, -0.22, 0.33, -0.41])        # cycles/sample in (-0.5, 0.5]
omegas_true = sigma + 1j * 2 * np.pi * f

N = len(a_true)

# Experimental grid
noise_levels = np.linspace(0.0, 0.5, num=10)
os_factors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Time vector: must be long enough for the largest oversampling factor
max_required_length = max(os_factors) * N + N + 1
t = np.arange(max_required_length)

num_trials = 1000  # Monte Carlo trials per configuration

# Output directory and file names
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = OUTPUT_DIR / "aggregated_results_with_condition.csv"
FREQ_TABLE_XLSX = OUTPUT_DIR / "frequency_error_table.xlsx"
AMP_TABLE_XLSX = OUTPUT_DIR / "amplitude_error_table.xlsx"
COMPARISON_XLSX = OUTPUT_DIR / "comparison_tables.xlsx"
FIGURE_RMSE_HEATMAP = OUTPUT_DIR / "figure_rmse_mean_heatmap.png"
FIGURE_RMSE_VS_RHO = OUTPUT_DIR / "figure_rmse_vs_rho_selected_sigmas.png"
FIGURE_COND_HEATMAP = OUTPUT_DIR / "figure_conditionnumber_hankel_heatmap.png"


def run_experiments() -> pd.DataFrame:
    """
    Run Monte Carlo experiments for each oversampling factor and noise level.

    For each configuration (os_factor, noise_level) the following are recorded:
        - Amplitude error (relative)
        - Frequency error (relative, with appropriate handling)
        - Backward error (full window and fixed window)
        - RMSE (full and fixed)
        - Condition number of the Hankel matrix
        - Error ratios (forward / backward)

    Returns
    -------
    df_results : pd.DataFrame
        Aggregated results (means and standard deviations) for each configuration.
    """
    y_clean = generate_clean_data(t, a_true, omegas_true)
    results = []

    total_start = time.perf_counter()

    for os_factor in os_factors:
        for noise_level in noise_levels:
            amp_errors = []
            freq_errors = []
            back_errors = []
            rel_back_errors = []
            cond_numbers = []
            amp_error_ratios = []
            freq_error_ratios = []
            rmse_list = []
            rel_rmse_list = []
            back_errors_fixed = []
            rel_back_errors_fixed = []
            rmse_fixed_list = []
            rel_rmse_fixed_list = []
            runtimes = []

            m = os_factor * N

            for trial in range(num_trials):
                rng = np.random.default_rng(trial)  # reproducible per trial

                y_noisy = generate_noisy_data(
                    y_clean[: m + N + 1],
                    noise_level,
                    rng=rng,
                )

                start_time = time.perf_counter()
                a_est, omegas_est, cond_num = prony_method(y_noisy, os_factor, n=N)
                runtimes.append(time.perf_counter() - start_time)

                a_est_m, omegas_est_m, _ = match_estimates(
                    a_true, omegas_true, a_est, omegas_est
                )

                amp_err = np.linalg.norm(a_true - a_est_m) / (np.linalg.norm(a_true) + 1e-15)
                amp_errors.append(amp_err)

                freq_err = frequency_error_generic(omegas_true, omegas_est_m)
                freq_errors.append(freq_err)

                cond_numbers.append(cond_num)

                back_err, rel_back_err, rmse, rel_rmse = compute_backward_error(
                    y_noisy,
                    a_est_m,
                    omegas_est_m,
                    oversampling_factor=os_factor,
                    N=N,
                )
                back_errors.append(back_err)
                rel_back_errors.append(rel_back_err)
                rmse_list.append(rmse)
                rel_rmse_list.append(rel_rmse)

                L_fixed = 2 * N + 1
                back_err_fx, rel_back_err_fx, rmse_fx, rel_rmse_fx = compute_backward_error(
                    y_noisy,
                    a_est_m,
                    omegas_est_m,
                    oversampling_factor=os_factor,
                    N=N,
                    L_eval=L_fixed,
                )
                back_errors_fixed.append(back_err_fx)
                rel_back_errors_fixed.append(rel_back_err_fx)
                rmse_fixed_list.append(rmse_fx)
                rel_rmse_fixed_list.append(rel_rmse_fx)

                if back_err != 0:
                    amp_error_ratios.append(amp_err / back_err)
                    freq_error_ratios.append(freq_err / back_err)
                else:
                    amp_error_ratios.append(np.nan)
                    freq_error_ratios.append(np.nan)

            results.append(
                {
                    "Oversampling Factor": os_factor,
                    "Noise Level": noise_level,
                    "Amplitude Error Mean": np.mean(amp_errors),
                    "Amplitude Error Std": np.std(amp_errors),
                    "Frequency Error Mean": np.mean(freq_errors),
                    "Frequency Error Std": np.std(freq_errors),
                    "Backward Error Mean": np.mean(back_errors),
                    "Backward Error Std": np.std(back_errors),
                    "Relative Backward Error Mean": np.mean(rel_back_errors),
                    "Relative Backward Error Std": np.std(rel_back_errors),
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
                    "Backward Error Fixed Mean": np.mean(back_errors_fixed),
                    "Backward Error Fixed Std": np.std(back_errors_fixed),
                    "Relative Backward Error Fixed Mean": np.mean(rel_back_errors_fixed),
                    "Relative Backward Error Fixed Std": np.std(rel_back_errors_fixed),
                    "RMSE Fixed Mean": np.mean(rmse_fixed_list),
                    "RMSE Fixed Std": np.std(rmse_fixed_list),
                    "Rel RMSE Fixed Mean": np.mean(rel_rmse_fixed_list),
                    "Rel RMSE Fixed Std": np.std(rel_rmse_fixed_list),
                    "Runtime Mean (s)": float(np.mean(runtimes)),
                    "Runtime Std (s)": float(np.std(runtimes)),
                }
            )

    total_time = time.perf_counter() - total_start
    print(f"Total experiment time: {total_time:.2f} seconds")
    print(f"Saving outputs to: {OUTPUT_DIR}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_CSV, index=False)
    return df_results


def plot_and_save_results(df_results: pd.DataFrame) -> None:
    """Generate all plots and save them (both static and interactive)."""
    table_frequency = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Frequency Error Mean"
    )
    table_amplitude = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Amplitude Error Mean"
    )

    print("\nFrequency error table (mean):")
    print(table_frequency.to_markdown())
    table_frequency.to_excel(FREQ_TABLE_XLSX, sheet_name="Frequency Errors")

    print("\nAmplitude error table (mean):")
    print(table_amplitude.to_markdown())
    table_amplitude.to_excel(AMP_TABLE_XLSX, sheet_name="Amplitude Errors")

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
    plt.show()

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
    plt.show()

    plot_error_heatmap(
        df_results,
        "Backward Error Mean",
        "Sampling Error Heatmap (Mean)\n(Noise Level vs Oversampling Factor)",
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
        save_path=FIGURE_RMSE_HEATMAP,
    )

    df_renamed = df_results.rename(
        columns={
            "Backward Error Mean": "Backward Error",
            "Relative Backward Error Mean": "Relative Backward Error",
        }
    )
    plot_error_subplots(df_renamed)

    plot_metric_vs_rho_selected_sigmas(
        df_results,
        "RMSE Mean",
        sigma_targets=[0.11, 0.22, 0.33, 0.50],
        title="RMSE vs Oversampling Factor (selected noise levels)",
        save_path=FIGURE_RMSE_VS_RHO,
    )

    grouped = df_results.groupby("Oversampling Factor").mean(numeric_only=True)
    amp_ratios_mean = grouped["Amplitude Error Ratio Mean"].values
    freq_ratios_mean = grouped["Frequency Error Ratio Mean"].values
    plot_error_ratios(os_factors, amp_ratios_mean, freq_ratios_mean)

    plot_condition_number_heatmap(
        df_results,
        title=r"Condition number of Hankel matrix ($\log_{10}\kappa(H)$)",
        save_path=FIGURE_COND_HEATMAP,
        drop_sigma0=True,
    )
    plot_condition_vs_error_scatter(df_results)
    plot_condition_number_by_noise(df_results)

    try:
        interactive_amplitude_error_plot(df_results)
        interactive_frequency_error_plot(df_results)
    except Exception as e:
        print("Interactive plotting encountered an error:", e)


def main() -> None:
    """Run the full experiment suite and produce all outputs."""
    df_results = run_experiments()
    plot_and_save_results(df_results)
    save_comparison_tables(df_results, excel_path=COMPARISON_XLSX)
    plot_sensitivity(a_true, omegas_true, t, os_factors)


if __name__ == "__main__":
    main()

