"""Reproducible n=6 and n=8 scope checks for the CMMAI study.

These experiments are deliberately separate from the main n=4 damped study.
They use purely imaginary exponents and therefore report wrapped frequency
RMSE (cycles/sample), rather than the main paper's relative complex-exponent
error. Amplitude error remains a relative l2 error.

The paired protocol matches the accepted-abstract main study: trial i uses
default_rng(i), and all rho values use progressively longer prefixes of the
same real Gaussian noise realization.

Run a smoke test:
    python -m experiments.paired_oversampling.scope_checks --trials 3

Run the full 1000-trial study:
    python -m experiments.paired_oversampling.scope_checks
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from prony import (
    match_estimates_wrapped_exponent,
    prony_method,
    wrap_to_nyquist,
)


from .noise_protocol import canonical_real_noise\n\nPROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "paired_oversampling"

RHO_VALUES = np.arange(1, 11, dtype=int)
NOISE_LEVELS = np.array([2.0 / 9.0, 0.5], dtype=float)
N_TRIALS = 1000
TRIAL_SEEDS = np.arange(N_TRIALS, dtype=int)

METRIC_NAMES = (
    "wrapped_frequency_rmse",
    "relative_amplitude_error",
)


@dataclass(frozen=True)
class ScopeModel:
    n: int
    amplitudes: np.ndarray
    frequencies: np.ndarray

    @property
    def exponents(self) -> np.ndarray:
        return 2.0j * np.pi * self.frequencies

    @property
    def max_samples(self) -> int:
        return self.n * (int(RHO_VALUES.max()) + 1) + 1

    def sample_count(self, rho: int) -> int:
        return self.n * (int(rho) + 1) + 1

    def clean_signal(self, length: int) -> np.ndarray:
        k = np.arange(length, dtype=float)
        V = np.exp(np.outer(k, self.exponents))
        return V @ self.amplitudes


MODELS = {
    6: ScopeModel(
        n=6,
        amplitudes=np.array(
            [1.0, 1.5, 0.8, 1.2, 0.9, 1.1],
            dtype=np.complex128,
        ),
        frequencies=np.array(
            [0.15, 0.33, 0.47, 0.61, 0.76, 0.92],
            dtype=float,
        ),
    ),
    8: ScopeModel(
        n=8,
        amplitudes=np.array(
            [1.0, 1.2, 0.7, 1.3, 0.6, 0.95, 1.1, 0.85],
            dtype=np.complex128,
        ),
        frequencies=np.array(
            [0.06, 0.18, 0.30, 0.42, 0.58, 0.70, 0.82, 0.94],
            dtype=float,
        ),
    ),
}


def _score_trial(
    model: ScopeModel,
    y_noisy: np.ndarray,
    rho: int,
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a_hat, lambda_hat, _ = prony_method(
            y_noisy,
            oversampling_factor=int(rho),
            n=model.n,
        )

    a_hat, lambda_hat, _ = match_estimates_wrapped_exponent(
        model.amplitudes,
        model.exponents,
        a_hat,
        lambda_hat,
    )

    f_hat = np.imag(lambda_hat) / (2.0 * np.pi)
    freq_diff = wrap_to_nyquist(f_hat - model.frequencies)
    frequency_rmse = float(np.sqrt(np.mean(freq_diff**2)))

    amplitude_error = float(
        np.linalg.norm(a_hat - model.amplitudes)
        / np.linalg.norm(model.amplitudes)
    )

    values = np.array(
        [frequency_rmse, amplitude_error],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(
            f"Non-finite scope-check metric for n={model.n}, rho={rho}: "
            f"{values}"
        )
    return values


def run(n_trials: int = N_TRIALS) -> dict[int, np.ndarray]:
    if not 1 <= n_trials <= N_TRIALS:
        raise ValueError(
            f"n_trials must be between 1 and {N_TRIALS}, got {n_trials}."
        )

    results: dict[int, np.ndarray] = {}

    for n, model in MODELS.items():
        y_clean_full = model.clean_signal(model.max_samples)
        raw = np.empty(
            (
                len(NOISE_LEVELS),
                len(RHO_VALUES),
                n_trials,
                len(METRIC_NAMES),
            ),
            dtype=float,
        )

        for noise_idx, sigma in enumerate(NOISE_LEVELS):
            print(
                f"n={n}: noise {noise_idx + 1}/{len(NOISE_LEVELS)} "
                f"sigma={sigma:.6f}",
                flush=True,
            )

            for trial_idx, seed in enumerate(TRIAL_SEEDS[:n_trials]):
                noise_full = canonical_real_noise(int(seed), model.max_samples)

                for rho_idx, rho in enumerate(RHO_VALUES):
                    n_samples = model.sample_count(int(rho))
                    y_noisy = (
                        y_clean_full[:n_samples]
                        + float(sigma) * noise_full[:n_samples]
                    )
                    raw[noise_idx, rho_idx, trial_idx, :] = _score_trial(
                        model,
                        y_noisy,
                        int(rho),
                    )

        results[n] = raw

    return results


def make_summary(results: dict[int, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for n, raw in results.items():
        for noise_idx, sigma in enumerate(NOISE_LEVELS):
            for rho_idx, rho in enumerate(RHO_VALUES):
                for metric_idx, metric_name in enumerate(METRIC_NAMES):
                    values = raw[noise_idx, rho_idx, :, metric_idx]
                    rows.append(
                        {
                            "n": int(n),
                            "noise_level": float(sigma),
                            "rho": int(rho),
                            "metric": metric_name,
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "median": float(np.median(values)),
                            "q25": float(np.quantile(values, 0.25)),
                            "q75": float(np.quantile(values, 0.75)),
                            "q95": float(np.quantile(values, 0.95)),
                        }
                    )

    return pd.DataFrame(rows)


def make_minima_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for n in sorted(MODELS):
        for sigma in NOISE_LEVELS:
            for metric in METRIC_NAMES:
                block = summary[
                    (summary["n"] == n)
                    & np.isclose(summary["noise_level"], sigma)
                    & (summary["metric"] == metric)
                ].sort_values("rho")

                best_idx = block["mean"].idxmin()
                best = block.loc[best_idx]
                rho1 = block[block["rho"] == 1].iloc[0]

                gain = (
                    100.0 * (float(rho1["mean"]) - float(best["mean"]))
                    / float(rho1["mean"])
                )

                rows.append(
                    {
                        "n": int(n),
                        "noise_level": float(sigma),
                        "metric": metric,
                        "best_rho": int(best["rho"]),
                        "rho1_mean": float(rho1["mean"]),
                        "best_mean": float(best["mean"]),
                        "gain_vs_rho1_pct": float(gain),
                    }
                )

    return pd.DataFrame(rows)


def save_results(
    results: dict[int, np.ndarray],
    n_trials: int,
) -> tuple[list[Path], Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if n_trials == N_TRIALS else f"_smoke_{n_trials}"

    raw_paths = []
    for n, raw in results.items():
        path = OUTPUT_DIR / f"scope_n{n}{suffix}.npz"
        model = MODELS[n]
        np.savez_compressed(
            path,
            raw=raw,
            metric_names=np.array(METRIC_NAMES),
            noise_levels=NOISE_LEVELS,
            rho_values=RHO_VALUES,
            trial_seeds=TRIAL_SEEDS[:n_trials],
            n=np.array(n),
            amplitudes=model.amplitudes,
            frequencies=model.frequencies,
            exponents=model.exponents,
        )
        raw_paths.append(path)

    summary = make_summary(results)
    summary_path = OUTPUT_DIR / f"scope_checks_summary{suffix}.csv"
    summary.to_csv(summary_path, index=False)

    minima = make_minima_table(summary)
    minima_path = OUTPUT_DIR / f"scope_checks_minima{suffix}.csv"
    minima.to_csv(minima_path, index=False)

    return raw_paths, summary_path, minima_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible n=6 and n=8 oversampling scope checks."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=N_TRIALS,
        help=f"Paired trials per noise level (default: {N_TRIALS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run(args.trials)
    raw_paths, summary_path, minima_path = save_results(
        results,
        args.trials,
    )

    print()
    for path in raw_paths:
        print(f"Saved raw scope results to: {path}")
    print(f"Saved scope summary to:     {summary_path}")
    print(f"Saved minima table to:      {minima_path}")

    minima = pd.read_csv(minima_path)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.precision", 6)
    print()
    print("SCOPE-CHECK MINIMA")
    print(minima.to_string(index=False))


if __name__ == "__main__":
    main()
