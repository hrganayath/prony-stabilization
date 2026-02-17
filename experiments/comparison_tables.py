from pathlib import Path
import pandas as pd
from typing import Union

def save_comparison_tables(
    df_results: pd.DataFrame,
    excel_path: Union[str, Path] = Path(__file__).resolve().parents[1] / "results" / "comparison_tables.xlsx"
) -> None:
    """
    Create and save comparison tables for key error metrics as an Excel file.

    The Excel file contains separate sheets for each metric's mean and standard
    deviation, including:
        - Frequency Error
        - Amplitude Error
        - Backward Error (full window)
        - Relative Backward Error (full window)
        - Condition Number of Hankel matrix
        - RMSE (full window)
        - Relative RMSE (full window)
        - Backward Error (fixed window, L = 2N+1)
        - Relative Backward Error (fixed window)
        - RMSE (fixed window)
        - Relative RMSE (fixed window)
        - Condition Number: min, max, median

    Parameters
    ----------
    df_results : pd.DataFrame
        Aggregated results from Monte Carlo experiments. Must contain columns
        with appropriate names (e.g., "Frequency Error Mean", "Frequency Error Std",
        "Backward Error Fixed Mean", etc.).
    excel_path : Union[str, Path], optional
        Path where the Excel file will be saved. Defaults to "results/comparison_tables.xlsx".
    """
    excel_path = Path(excel_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    # Frequency Error
    table_frequency_mean = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Frequency Error Mean"
    )
    table_frequency_std = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Frequency Error Std"
    )

    # Amplitude Error
    table_amplitude_mean = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Amplitude Error Mean"
    )
    table_amplitude_std = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Amplitude Error Std"
    )

    # Backward Error (full window)
    table_backward_mean = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Backward Error Mean"
    )
    table_backward_std = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Backward Error Std"
    )

    # Relative Backward Error (full window)
    table_rel_backward_mean = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Relative Backward Error Mean"
    )
    table_rel_backward_std = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Relative Backward Error Std"
    )

    # Condition Number
    table_condition_mean = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Condition Number Mean"
    )
    table_condition_std = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Condition Number Std"
    )

    # RMSE (full window)
    table_rmse_mean = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="RMSE Mean"
    )
    table_rmse_std = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="RMSE Std"
    )

    # Relative RMSE (full window)
    table_rel_rmse_mean = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Rel RMSE Mean"
    )
    table_rel_rmse_std = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Rel RMSE Std"
    )

    # Backward Error (fixed window)
    table_backward_fixed_mean = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Backward Error Fixed Mean"
    )
    table_backward_fixed_std = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Backward Error Fixed Std"
    )

    # Relative Backward Error (fixed window)
    table_rel_backward_fixed_mean = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Relative Backward Error Fixed Mean"
    )
    table_rel_backward_fixed_std = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Relative Backward Error Fixed Std"
    )

    # RMSE (fixed window)
    table_rmse_fixed_mean = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="RMSE Fixed Mean"
    )
    table_rmse_fixed_std = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="RMSE Fixed Std"
    )

    # Relative RMSE (fixed window)
    table_rel_rmse_fixed_mean = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Rel RMSE Fixed Mean"
    )
    table_rel_rmse_fixed_std = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Rel RMSE Fixed Std"
    )

    # Additional condition number statistics
    table_condition_min = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Condition Number Min"
    )
    table_condition_max = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Condition Number Max"
    )
    table_condition_median = df_results.pivot(
        index="Noise Level", columns="Oversampling Factor", values="Condition Number Median"
    )

    # Write all sheets to Excel
    with pd.ExcelWriter(excel_path) as writer:
        table_frequency_mean.to_excel(writer, sheet_name="Freq Error Mean")
        table_frequency_std.to_excel(writer, sheet_name="Freq Error Std")
        table_amplitude_mean.to_excel(writer, sheet_name="Amp Error Mean")
        table_amplitude_std.to_excel(writer, sheet_name="Amp Error Std")
        table_backward_mean.to_excel(writer, sheet_name="Backward Error Mean")
        table_backward_std.to_excel(writer, sheet_name="Backward Error Std")
        table_rel_backward_mean.to_excel(writer, sheet_name="Rel Back Error Mean")
        table_rel_backward_std.to_excel(writer, sheet_name="Rel Back Error Std")
        table_condition_mean.to_excel(writer, sheet_name="Cond Num Mean")
        table_condition_std.to_excel(writer, sheet_name="Cond Num Std")
        table_rmse_mean.to_excel(writer, sheet_name="RMSE Mean")
        table_rmse_std.to_excel(writer, sheet_name="RMSE Std")
        table_rel_rmse_mean.to_excel(writer, sheet_name="Rel RMSE Mean")
        table_rel_rmse_std.to_excel(writer, sheet_name="Rel RMSE Std")

        table_backward_fixed_mean.to_excel(writer, sheet_name="Backward Fixed Mean")
        table_backward_fixed_std.to_excel(writer, sheet_name="Backward Fixed Std")
        table_rel_backward_fixed_mean.to_excel(writer, sheet_name="Rel Back Fixed Mean")
        table_rel_backward_fixed_std.to_excel(writer, sheet_name="Rel Back Fixed Std")

        table_rmse_fixed_mean.to_excel(writer, sheet_name="RMSE Fixed Mean")
        table_rmse_fixed_std.to_excel(writer, sheet_name="RMSE Fixed Std")
        table_rel_rmse_fixed_mean.to_excel(writer, sheet_name="Rel RMSE Fixed Mean")
        table_rel_rmse_fixed_std.to_excel(writer, sheet_name="Rel RMSE Fixed Std")

        table_condition_min.to_excel(writer, sheet_name="Cond Num Min")
        table_condition_max.to_excel(writer, sheet_name="Cond Num Max")
        table_condition_median.to_excel(writer, sheet_name="Cond Num Median")

    print(f"Comparison tables saved to {excel_path}")
