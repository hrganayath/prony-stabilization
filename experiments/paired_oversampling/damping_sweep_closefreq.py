"""Additional damped n=4 frequency-configuration control for CMMAI 2026.

Purpose
-------
Test whether the damping-dependent shift in the useful rho range persists
for a second, more closely spaced frequency configuration.

Everything is kept the same as in the original damping sweep except for
the frequencies.

Original amplitudes and damping vector are retained.
New frequencies are deliberately closer together:
    f = (0.10, 0.16, 0.22, 0.28)

Damping scales:
    alpha in {0.0, 0.5, 1.0, 1.5}

Noise levels:
    sigma in {2/9, 0.5}

Oversampling factors:
    rho = 1,...,10

Trials:
    1000 paired trials using the canonical real-noise protocol.

Run smoke test:
    python3.10 -m experiments.paired_oversampling.damping_sweep_closefreq --trials 3

Run full:
    python3.10 -m experiments.paired_oversampling.damping_sweep_closefreq
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from prony import (
    match_estimates_wrapped_exponent,
    prony_method,
    relative_exponent_error_wrapped,
    wrap_to_nyquist,
)

from .config import AMPLITUDES, EXPONENTS, MODEL_ORDER, RHO_VALUES
from .noise_protocol import canonical_real_noise

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "paired_oversampling"

DAMPING_SCALES = np.array([0.0, 0.5, 1.0, 1.5], dtype=float)
NOISE_LEVELS = np.array([2.0 / 9.0, 0.5], dtype=float)

N_TRIALS = 1000
TRIAL_SEEDS = np.arange(N_TRIALS, dtype=int)

MAX_RHO = int(np.max(RHO_VALUES))
MAX_SAMPLES = MODEL_ORDER * (MAX_RHO + 1) + 1

BASE_DAMPING = np.real(EXPONENTS).copy()

# Second n=4 configuration: same amplitudes, closer frequency spacing.
ALT_FREQUENCIES = np.array([0.08, -0.18, 0.29, -0.38], dtype=float)

METRIC_NAMES = (
    "wrapped_exponent_error",
    "relative_amplitude_error",
    "damping_rmse",
    "wrapped_frequency_rmse",
)


def scaled_exponents(alpha: float) -> np.ndarray:
    damping = float(alpha) * BASE_DAMPING
    return damping + 2.0j * np.pi * ALT_FREQUENCIES


def clean_signal(exponents: np.ndarray, length: int) -> np.ndarray:
    k = np.arange(length, dtype=float)
    return np.exp(np.outer(k, exponents)) @ AMPLITUDES


def sample_count(rho: int) -> int:
    return MODEL_ORDER * (int(rho) + 1) + 1


def score_trial(y_noisy: np.ndarray, rho: int, true_exponents: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a_hat, w_hat, _ = prony_method(
            y_noisy,
            oversampling_factor=int(rho),
            n=MODEL_ORDER,
        )

    a_hat, w_hat, _ = match_estimates_wrapped_exponent(
        AMPLITUDES,
        true_exponents,
        a_hat,
        w_hat,
    )

    exponent_error = float(
        relative_exponent_error_wrapped(true_exponents, w_hat)
    )

    amplitude_error = float(
        np.linalg.norm(a_hat - AMPLITUDES)
        / np.linalg.norm(AMPLITUDES)
    )

    damping_rmse = float(
        np.sqrt(
            np.mean(
                (np.real(w_hat) - np.real(true_exponents)) ** 2
            )
        )
    )

    f_true = np.imag(true_exponents) / (2.0 * np.pi)
    f_hat = np.imag(w_hat) / (2.0 * np.pi)
    df = wrap_to_nyquist(f_hat - f_true)
    frequency_rmse = float(np.sqrt(np.mean(df**2)))

    out = np.array(
        [exponent_error, amplitude_error, damping_rmse, frequency_rmse],
        dtype=float,
    )

    if not np.all(np.isfinite(out)):
        raise FloatingPointError(
            f"Non-finite metric for rho={rho}: {out}"
        )

    return out


def run(n_trials: int = N_TRIALS):
    if not 1 <= n_trials <= N_TRIALS:
        raise ValueError("invalid trial count")

    raw = np.empty(
        (
            len(DAMPING_SCALES),
            len(NOISE_LEVELS),
            len(RHO_VALUES),
            n_trials,
            len(METRIC_NAMES),
        ),
        dtype=float,
    )

    for ai, alpha in enumerate(DAMPING_SCALES):
        true_exponents = scaled_exponents(float(alpha))
        clean_full = clean_signal(true_exponents, MAX_SAMPLES)

        for si, sigma in enumerate(NOISE_LEVELS):
            print(
                f"alpha={alpha:.1f}: noise {si+1}/{len(NOISE_LEVELS)} "
                f"sigma={sigma:.6f}",
                flush=True,
            )

            for ti, seed in enumerate(TRIAL_SEEDS[:n_trials]):
                noise_full = canonical_real_noise(int(seed), MAX_SAMPLES)

                for ri, rho in enumerate(RHO_VALUES):
                    N = sample_count(int(rho))
                    y = clean_full[:N] + float(sigma) * noise_full[:N]

                    raw[ai, si, ri, ti, :] = score_trial(
                        y_noisy=y,
                        rho=int(rho),
                        true_exponents=true_exponents,
                    )

    return raw


def make_summary(raw):
    rows = []

    for ai, alpha in enumerate(DAMPING_SCALES):
        for si, sigma in enumerate(NOISE_LEVELS):
            for ri, rho in enumerate(RHO_VALUES):
                for mi, metric in enumerate(METRIC_NAMES):
                    v = raw[ai, si, ri, :, mi]
                    rows.append(
                        {
                            "damping_scale": float(alpha),
                            "noise_level": float(sigma),
                            "rho": int(rho),
                            "metric": metric,
                            "mean": float(np.mean(v)),
                            "std": float(np.std(v)),
                            "median": float(np.median(v)),
                            "q25": float(np.quantile(v, 0.25)),
                            "q75": float(np.quantile(v, 0.75)),
                            "q95": float(np.quantile(v, 0.95)),
                        }
                    )

    return pd.DataFrame(rows)


def make_minima(summary):
    rows = []

    for alpha in DAMPING_SCALES:
        for sigma in NOISE_LEVELS:
            for metric in METRIC_NAMES:
                block = summary[
                    np.isclose(summary["damping_scale"], alpha)
                    & np.isclose(summary["noise_level"], sigma)
                    & (summary["metric"] == metric)
                ].sort_values("rho")

                best = block.loc[block["mean"].idxmin()]
                rho1 = block[block["rho"] == 1].iloc[0]

                rows.append(
                    {
                        "damping_scale": float(alpha),
                        "noise_level": float(sigma),
                        "metric": metric,
                        "best_rho": int(best["rho"]),
                        "rho1_mean": float(rho1["mean"]),
                        "best_mean": float(best["mean"]),
                        "gain_vs_rho1_pct": float(
                            100.0
                            * (rho1["mean"] - best["mean"])
                            / rho1["mean"]
                        ),
                    }
                )

    return pd.DataFrame(rows)


def save(raw, n_trials):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suffix = "" if n_trials == N_TRIALS else f"_smoke_{n_trials}"

    raw_path = OUTPUT_DIR / f"damping_sweep_closefreq{suffix}.npz"
    summary_path = OUTPUT_DIR / f"damping_sweep_closefreq_summary{suffix}.csv"
    minima_path = OUTPUT_DIR / f"damping_sweep_closefreq_minima{suffix}.csv"

    np.savez_compressed(
        raw_path,
        raw=raw,
        metric_names=np.array(METRIC_NAMES),
        damping_scales=DAMPING_SCALES,
        noise_levels=NOISE_LEVELS,
        rho_values=RHO_VALUES,
        trial_seeds=TRIAL_SEEDS[:n_trials],
        amplitudes=AMPLITUDES,
        base_damping=BASE_DAMPING,
        frequencies=ALT_FREQUENCIES,
    )

    summary = make_summary(raw)
    summary.to_csv(summary_path, index=False)

    minima = make_minima(summary)
    minima.to_csv(minima_path, index=False)

    return raw_path, summary_path, minima_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    raw = run(args.trials)
    raw_path, summary_path, minima_path = save(raw, args.trials)

    print()
    print(f"Saved raw close-frequency damping sweep to: {raw_path}")
    print(f"Saved summary to:                         {summary_path}")
    print(f"Saved minima table to:                    {minima_path}")

    minima = pd.read_csv(minima_path)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.precision", 6)

    print()
    print("CLOSE-FREQUENCY DAMPING-SWEEP MINIMA")
    print(minima.to_string(index=False))


if __name__ == "__main__":
    main()
