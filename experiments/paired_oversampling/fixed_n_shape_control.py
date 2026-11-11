"""Fixed-N Hankel-shape control for the CMMAI oversampling study."""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import hankel, svd

from prony import (
    match_estimates_wrapped_exponent,
    relative_exponent_error_wrapped,
    wrap_to_nyquist,
)

from .config import AMPLITUDES, EXPONENTS, MODEL_ORDER
from .noise_protocol import canonical_real_noise

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "paired_oversampling"

N_VALUES = np.array([17, 29, 45], dtype=int)
NOISE_LEVELS = np.array([2.0 / 9.0, 0.5], dtype=float)
N_TRIALS = 1000
TRIAL_SEEDS = np.arange(N_TRIALS, dtype=int)

METRIC_NAMES = (
    "wrapped_exponent_error",
    "relative_amplitude_error",
    "damping_rmse",
    "wrapped_frequency_rmse",
    "hankel_condition",
)


def clean_signal(length):
    k = np.arange(length, dtype=float)
    return np.exp(np.outer(k, EXPONENTS)) @ AMPLITUDES


def column_grid(N):
    cmin = MODEL_ORDER + 1
    cmax = (int(N) + 1) // 2
    vals = np.arange(cmin, cmax + 1, 2, dtype=int)

    if vals[-1] != cmax:
        vals = np.unique(np.append(vals, cmax)).astype(int)

    return vals


def fit_with_shape(y, n, columns):
    N = len(y)
    C = int(columns)
    R = N - C + 1

    if C < n + 1 or R < n + 1:
        raise ValueError("Hankel dimensions too small for model order")

    H = hankel(y[:R], y[R - 1 : N])
    cond = float(np.linalg.cond(H))

    U, S, _ = svd(H, full_matrices=False)
    Us = U[:, :n]

    Phi, _, _, _ = np.linalg.lstsq(
        Us[:-1, :],
        Us[1:, :],
        rcond=None,
    )
    z, _ = np.linalg.eig(Phi)
    z = z.astype(complex)

    if np.any(np.abs(z) < np.finfo(float).eps * 100):
        raise ValueError("Estimated pole near zero")

    w = np.log(z)
    t = np.arange(N, dtype=float)
    V = np.exp(np.outer(t, w))

    norms = np.linalg.norm(V, axis=0)
    threshold = np.finfo(float).eps * np.max(norms)
    norms[norms < threshold] = 1.0

    Vs = V / norms
    asc, _, _, _ = np.linalg.lstsq(Vs, y, rcond=None)
    a = asc / norms

    return a, w, cond


def score_trial(y, columns):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a, w, cond = fit_with_shape(y, MODEL_ORDER, columns)

    a, w, _ = match_estimates_wrapped_exponent(
        AMPLITUDES,
        EXPONENTS,
        a,
        w,
    )

    eexp = float(relative_exponent_error_wrapped(EXPONENTS, w))
    eamp = float(
        np.linalg.norm(a - AMPLITUDES) / np.linalg.norm(AMPLITUDES)
    )
    edamp = float(
        np.sqrt(np.mean((np.real(w) - np.real(EXPONENTS)) ** 2))
    )

    ftrue = np.imag(EXPONENTS) / (2 * np.pi)
    fhat = np.imag(w) / (2 * np.pi)
    df = wrap_to_nyquist(fhat - ftrue)
    efreq = float(np.sqrt(np.mean(df**2)))

    out = np.array([eexp, eamp, edamp, efreq, cond], dtype=float)

    if not np.all(np.isfinite(out)):
        raise FloatingPointError(
            f"Non-finite metric N={len(y)}, C={columns}"
        )

    return out


def run(n_trials=N_TRIALS):
    if not 1 <= n_trials <= N_TRIALS:
        raise ValueError("invalid trial count")

    maxN = int(N_VALUES.max())
    clean = clean_signal(maxN)
    results = {}

    for N in N_VALUES:
        Cs = column_grid(int(N))
        raw = np.empty(
            (
                len(NOISE_LEVELS),
                len(Cs),
                n_trials,
                len(METRIC_NAMES),
            ),
            dtype=float,
        )

        for si, sigma in enumerate(NOISE_LEVELS):
            print(
                f"N={N}: noise {si + 1}/{len(NOISE_LEVELS)} "
                f"sigma={sigma:.6f}",
                flush=True,
            )

            for ti, seed in enumerate(TRIAL_SEEDS[:n_trials]):
                noise = canonical_real_noise(int(seed), maxN)
                y = clean[:N] + float(sigma) * noise[:N]

                for ci, C in enumerate(Cs):
                    raw[si, ci, ti, :] = score_trial(y, int(C))

        results[int(N)] = raw

    return results


def make_summary(results):
    rows = []

    for N, raw in results.items():
        Cs = column_grid(N)

        for si, sigma in enumerate(NOISE_LEVELS):
            for ci, C in enumerate(Cs):
                R = N - int(C) + 1

                for mi, name in enumerate(METRIC_NAMES):
                    v = raw[si, ci, :, mi]
                    rows.append(
                        {
                            "N": N,
                            "noise_level": float(sigma),
                            "columns": int(C),
                            "rows": R,
                            "aspect_rows_over_columns": R / float(C),
                            "metric": name,
                            "mean": float(np.mean(v)),
                            "std": float(np.std(v)),
                            "median": float(np.median(v)),
                            "q25": float(np.quantile(v, 0.25)),
                            "q75": float(np.quantile(v, 0.75)),
                            "q95": float(np.quantile(v, 0.95)),
                        }
                    )

    return pd.DataFrame(rows)


def make_comparison(summary):
    rows = []
    metrics = (
        "wrapped_exponent_error",
        "relative_amplitude_error",
        "damping_rmse",
        "wrapped_frequency_rmse",
    )

    for N in N_VALUES:
        Cs = column_grid(int(N))
        bal = int(Cs[-1])

        for sigma in NOISE_LEVELS:
            for metric in metrics:
                b = summary[
                    (summary.N == int(N))
                    & np.isclose(summary.noise_level, sigma)
                    & (summary.metric == metric)
                ].sort_values("columns")

                narrow = b[b["columns"] == MODEL_ORDER + 1].iloc[0]
                balanced = b[b["columns"] == bal].iloc[0]
                best = b.loc[b["mean"].idxmin()]

                rows.append(
                    {
                        "N": int(N),
                        "noise_level": float(sigma),
                        "metric": metric,
                        "narrow_columns": MODEL_ORDER + 1,
                        "narrow_mean": float(narrow["mean"]),
                        "best_columns": int(best["columns"]),
                        "best_rows": int(best["rows"]),
                        "best_mean": float(best["mean"]),
                        "gain_best_vs_narrow_pct": (
                            100
                            * (float(narrow["mean"]) - float(best["mean"]))
                            / float(narrow["mean"])
                        ),
                        "balanced_columns": bal,
                        "balanced_rows": int(N - bal + 1),
                        "balanced_mean": float(balanced["mean"]),
                        "gain_balanced_vs_narrow_pct": (
                            100
                            * (
                                float(narrow["mean"])
                                - float(balanced["mean"])
                            )
                            / float(narrow["mean"])
                        ),
                    }
                )

    return pd.DataFrame(rows)


def save(results, n_trials):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if n_trials == N_TRIALS else f"_smoke_{n_trials}"

    for N, raw in results.items():
        np.savez_compressed(
            OUTPUT_DIR / f"fixed_n_shape_N{N}{suffix}.npz",
            raw=raw,
            metric_names=np.array(METRIC_NAMES),
            noise_levels=NOISE_LEVELS,
            columns=column_grid(N),
            trial_seeds=TRIAL_SEEDS[:n_trials],
            N=np.array(N),
        )

    summary = make_summary(results)
    sp = OUTPUT_DIR / f"fixed_n_shape_summary{suffix}.csv"
    summary.to_csv(sp, index=False)

    comp = make_comparison(summary)
    cp = OUTPUT_DIR / f"fixed_n_shape_comparison{suffix}.csv"
    comp.to_csv(cp, index=False)

    return sp, cp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    args = ap.parse_args()

    results = run(args.trials)
    sp, cp = save(results, args.trials)

    print(
        f"\nSaved fixed-N summary to: {sp}\n"
        f"Saved comparison table to: {cp}\n"
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.precision", 6)

    print(pd.read_csv(cp).to_string(index=False))


if __name__ == "__main__":
    main()
