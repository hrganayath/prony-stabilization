import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.patheffects as pe
from typing import Optional, List

# ----------------------------------------------------------------------
# Helper for consistent label mapping (optional)
_LABEL_MAP = {
    "Residual Mean": "Residual norm (mean)",
    "Relative Residual Mean": "Relative residual (mean)",
    "RMSE Mean": "RMSE Mean",
    "Rel RMSE Mean": "Relative RMSE Mean",
    # add more if needed
}

def _get_pretty_label(value_col: str) -> str:
    """Return a pretty axis label for a given column name."""
    return _LABEL_MAP.get(value_col, value_col.replace("_", " "))

# ----------------------------------------------------------------------

def plot_metric_vs_rho_selected_sigmas(
    df_results: pd.DataFrame,
    value_col: str,
    sigma_targets: List[float],
    title: str,
    save_path: Optional[str] = None
) -> None:
    """
    Plot a given metric (e.g., RMSE) vs oversampling factor for selected noise levels.

    Parameters
    ----------
    df_results : pd.DataFrame
        Aggregated results with columns "Noise Level", "Oversampling Factor", and value_col.
    value_col : str
        Column name of the metric to plot.
    sigma_targets : list of float
        Target noise levels; the closest available level in df_results is used.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    df = df_results.copy()
    df["Oversampling Factor"] = df["Oversampling Factor"].astype(int)

    sigmas_avail = np.sort(df["Noise Level"].unique())

    plt.figure(figsize=(9, 5))
    for s in sigma_targets:
        s_used = sigmas_avail[np.argmin(np.abs(sigmas_avail - s))]
        sub = df[df["Noise Level"] == s_used].sort_values("Oversampling Factor")
        plt.plot(sub["Oversampling Factor"], sub[value_col], marker="o", label=f"σ≈{s_used:.2f}")

    plt.title(title)
    plt.xlabel("Oversampling Factor (ρ)")
    plt.ylabel(_get_pretty_label(value_col))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_error_heatmap(
    df_results: pd.DataFrame,
    value_col: str,
    title: str,
    save_path: Optional[str] = None
) -> None:
    """
    Create a heatmap of error values (noise level vs oversampling factor).

    Parameters
    ----------
    df_results : pd.DataFrame
        Aggregated results with columns "Noise Level", "Oversampling Factor", and value_col.
    value_col : str
        Column name of the error metric to plot.
    title : str
        Heatmap title.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    df = df_results.copy()
    df["Oversampling Factor"] = df["Oversampling Factor"].astype(int)

    heatmap_data = df.pivot(
        index="Noise Level",
        columns="Oversampling Factor",
        values=value_col
    )

    pretty_label = _get_pretty_label(value_col)

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2e",
        cmap="YlGnBu",
        annot_kws={"size": 9},
        linewidths=0.5,
        cbar_kws={"label": pretty_label}
    )
    ax.set_title(title)
    ax.set_xlabel("Oversampling Factor (ρ)")
    ax.set_ylabel("Noise Level (σ)")

    # Nicer tick labels
    ax.set_yticklabels([f"{float(v):.2f}" for v in heatmap_data.index], rotation=0)
    ax.set_xticklabels([str(int(v)) for v in heatmap_data.columns], rotation=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_error_subplots(
    df_results: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Create subplots for absolute and relative backward error vs noise level.

    Parameters
    ----------
    df_results : pd.DataFrame
        Aggregated results. Must contain columns "Oversampling Factor", "Noise Level",
        "Residual" (absolute) and "Relative Residual".
    save_path : str, optional
        If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Absolute backward error
    for os_factor in sorted(df_results["Oversampling Factor"].unique()):
        subset = df_results[df_results["Oversampling Factor"] == os_factor]
        ax[0].plot(subset["Noise Level"], subset["Residual"], label=f'ρ = {os_factor}')
    ax[0].set_title("Absolute Residual norm vs Noise Level")
    ax[0].set_ylabel("Residual norm")
    ax[0].legend()

    # Relative backward error
    for os_factor in sorted(df_results["Oversampling Factor"].unique()):
        subset = df_results[df_results["Oversampling Factor"] == os_factor]
        ax[1].plot(subset["Noise Level"], subset["Relative Residual"], label=f'ρ = {os_factor}')
    ax[1].set_title("Relative Residual norm vs Noise Level")
    ax[1].set_xlabel("Noise Level")
    ax[1].set_ylabel("Relative Residual norm")
    ax[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def interactive_amplitude_error_plot(
    df_results: pd.DataFrame,
    keep_os_factors: Optional[List[int]] = None
) -> None:
    """
    Create an interactive Plotly line plot of amplitude error vs noise level
    for selected oversampling factors.

    Parameters
    ----------
    df_results : pd.DataFrame
        Aggregated results with columns "Noise Level", "Oversampling Factor",
        "Amplitude Error Mean".
    keep_os_factors : list of int, optional
        Oversampling factors to include. Default [1, 3, 5, 8, 10].
    """
    if keep_os_factors is None:
        keep_os_factors = [1, 3, 5, 8, 10]
    keep = set(keep_os_factors)

    df = df_results.copy()
    df["Oversampling Factor"] = df["Oversampling Factor"].astype(int)
    df = df[df["Oversampling Factor"].isin(keep)]
    df["Oversampling Factor"] = pd.Categorical(
        df["Oversampling Factor"], categories=keep_os_factors, ordered=True
    )

    fig = px.line(
        df,
        x="Noise Level",
        y="Amplitude Error Mean",
        color="Oversampling Factor",
        markers=True,
        title="Amplitude Error vs Noise Level",
        labels={
            "Noise Level": "Noise Level (σ)",
            "Amplitude Error Mean": "Amplitude Error (Mean)",
            "Oversampling Factor": "ρ",
        },
        template="simple_white"
    )

    fig.update_layout(
        title_x=0.5,
        font=dict(size=14, color="black"),
        title_font=dict(size=22, color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title_text="ρ",
        margin=dict(l=60, r=40, t=60, b=60),
    )
    fig.update_traces(line=dict(width=1.8), marker=dict(size=6))
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.35)",
        showline=True, linecolor="black", tickfont=dict(size=12, color="black"),
        title_font=dict(size=16, color="black")
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.35)",
        showline=True, linecolor="black", tickfont=dict(size=12, color="black"),
        title_font=dict(size=16, color="black")
    )
    fig.update_traces(marker=dict(size=8))

    fig.show()


def interactive_frequency_error_plot(
    df_results: pd.DataFrame,
    keep_os_factors: Optional[List[int]] = None
) -> None:
    """
    Create an interactive Plotly line plot of frequency error vs noise level
    for selected oversampling factors.

    Parameters
    ----------
    df_results : pd.DataFrame
        Aggregated results with columns "Noise Level", "Oversampling Factor",
        "Frequency Error Mean".
    keep_os_factors : list of int, optional
        Oversampling factors to include. Default [1, 3, 5, 8, 10].
    """
    if keep_os_factors is None:
        keep_os_factors = [1, 3, 5, 8, 10]
    keep = set(keep_os_factors)

    df = df_results.copy()
    df["Oversampling Factor"] = df["Oversampling Factor"].astype(int)
    df = df[df["Oversampling Factor"].isin(keep)]
    df["Oversampling Factor"] = pd.Categorical(
        df["Oversampling Factor"], categories=keep_os_factors, ordered=True
    )

    fig = px.line(
        df,
        x="Noise Level",
        y="Frequency Error Mean",
        color="Oversampling Factor",
        markers=True,
        title="Exponent Error vs Noise Level",
        labels={
            "Noise Level": "Noise Level (σ)",
            "Frequency Error Mean": "Exponent Error (Mean)",
            "Oversampling Factor": "ρ",
        },
        template="simple_white"
    )

    fig.update_layout(
        title_x=0.5,
        font=dict(size=14, color="black"),
        title_font=dict(size=22, color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title_text="ρ",
        margin=dict(l=60, r=40, t=60, b=60),
    )
    fig.update_traces(line=dict(width=1.8), marker=dict(size=6))
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.35)",
        showline=True, linecolor="black", tickfont=dict(size=12, color="black"),
        title_font=dict(size=16, color="black")
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.35)",
        showline=True, linecolor="black", tickfont=dict(size=12, color="black"),
        title_font=dict(size=16, color="black")
    )
    fig.update_traces(marker=dict(size=8))

    fig.show()


def plot_error_ratios(
    os_factors: List[int],
    amp_ratios_mean: np.ndarray,
    freq_ratios_mean: np.ndarray,
    title: str = "Error Ratio vs Oversampling Factor",
    save_path: Optional[str] = None
) -> None:
    """
    Bar plot of mean error ratios (forward error / residual) vs oversampling factor.

    Parameters
    ----------
    os_factors : list of int
        Oversampling factors (x‑axis).
    amp_ratios_mean : np.ndarray
        Mean amplitude error ratios for each oversampling factor.
    freq_ratios_mean : np.ndarray
        Mean frequency error ratios for each oversampling factor.
    title : str, optional
        Plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    plt.figure(figsize=(8, 6))
    plt.yscale('log')
    plt.ylabel('Log Mean Error Ratio (Forward Error / Residual norm)')

    width = 0.4
    x = np.arange(len(os_factors))

    plt.bar(x + width/2, freq_ratios_mean, width, label='Exponent Error Ratio', color='tab:orange')
    plt.bar(x - width/2, amp_ratios_mean, width, label='Amplitude Error Ratio', color='tab:blue')

    plt.xticks(x, os_factors)
    plt.xlabel('Oversampling Factor')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_condition_number_by_noise(
    df_results: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot condition number vs oversampling factor for each noise level.

    Parameters
    ----------
    df_results : pd.DataFrame
        Aggregated results with columns "Noise Level", "Oversampling Factor",
        "Condition Number Mean".
    save_path : str, optional
        If provided, save the figure to this path.
    """
    plt.figure(figsize=(10, 6))
    for noise_level in sorted(df_results["Noise Level"].unique()):
        subset = df_results[df_results["Noise Level"] == noise_level]
        plt.plot(subset["Oversampling Factor"], subset["Condition Number Mean"],
                 marker='o', label=f'Noise = {noise_level:.2f}')
    plt.yscale('log')
    plt.xlabel("Oversampling Factor")
    plt.ylabel("Condition Number (log scale)")
    plt.title("Condition Number of Hankel Matrix vs Oversampling\nOne Line per Noise Level")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_condition_number_heatmap(
    df_results: pd.DataFrame,
    title: str,
    save_path: Optional[str] = None,
    drop_sigma0: bool = True
) -> None:
    """
    Heatmap of log10 condition number of Hankel matrix (noise level vs oversampling factor).

    Parameters
    ----------
    df_results : pd.DataFrame
        Aggregated results with columns "Noise Level", "Oversampling Factor",
        "Condition Number Mean".
    title : str
        Plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    drop_sigma0 : bool, default True
        If True, exclude rows with noise level 0 to avoid extreme values dominating the colormap.
    """
    df = df_results.copy()
    df["Oversampling Factor"] = df["Oversampling Factor"].astype(int)

    heatmap_data = df.pivot(
        index="Noise Level",
        columns="Oversampling Factor",
        values="Condition Number Mean"
    )

    if drop_sigma0:
        heatmap_data = heatmap_data.loc[heatmap_data.index > 0.0]

    # Plot log10 of condition number
    log_data = np.log10(heatmap_data)

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        log_data,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        annot_kws={"size": 9},
        linewidths=0.5,
        cbar_kws={"label": r"$\log_{10}\kappa(H)$"}
    )

    # Make annotations readable
    for t in ax.texts:
        t.set_color("white")
        t.set_path_effects([pe.Stroke(linewidth=1.5, foreground="black"), pe.Normal()])

    ax.set_title(title)
    ax.set_xlabel("Oversampling Factor (ρ)")
    ax.set_ylabel("Noise Level (σ)")

    ax.set_yticklabels([f"{float(v):.2f}" for v in log_data.index], rotation=0)
    ax.set_xticklabels([str(int(v)) for v in log_data.columns], rotation=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_condition_vs_error_scatter(
    df_results: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Scatter plots showing correlation between condition number and errors.

    Parameters
    ----------
    df_results : pd.DataFrame
        Aggregated results with columns "Condition Number Mean",
        "Residual Mean", "Amplitude Error Mean".
    save_path : str, optional
        If provided, save the figure to this path.
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(df_results["Condition Number Mean"], df_results["Residual Mean"])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Condition Number (log scale)")
    plt.ylabel("Residual norm")
    plt.title("κ(H) vs Residual norm")

    plt.subplot(1, 2, 2)
    plt.scatter(df_results["Condition Number Mean"], df_results["Amplitude Error Mean"])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Condition Number (log scale)")
    plt.ylabel("Amplitude Error")
    plt.title("κ(H) vs Amplitude Error")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
