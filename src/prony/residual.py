# residual.py
import numpy as np
from typing import Optional, Tuple

from .data import generate_clean_data


def compute_residual_error(
    y_noisy: np.ndarray,
    a_hat: np.ndarray,
    omega_hat: np.ndarray,
    oversampling_factor: int,
    N: int,
    L_eval: Optional[int] = None,
    eps: float = 1e-15,
) -> Tuple[float, float, float, float]:
    """
    Compute residual (data misfit) metrics between a reconstructed signal and observed samples.

    We reconstruct y_hat using the estimated parameters (a_hat, omega_hat) and compare it to
    the observed samples y_ref = y_noisy[:L_eval].

    Metrics returned
    ----------------
    residual_norm:
        || y_hat - y_ref ||_2
    relative_residual:
        residual_norm / (||y_ref||_2 + eps)
    rmse:
        residual_norm / sqrt(L_eval)
    rel_rmse:
        rmse / ( (||y_ref||_2 / sqrt(L_eval)) + eps )

    Notes
    -----
    - This is a fit residual / data misfit, not a classical numerical-analysis backward error.
    - If L_eval is None, we evaluate on the full estimation window
      L_train = oversampling_factor * N + N + 1, which changes with oversampling_factor.
      For fair cross-factor comparisons, pass a fixed L_eval (e.g., 2*N+1).
    """
    L_train = oversampling_factor * N + N + 1

    if L_eval is None:
        L_eval = L_train
    else:
        L_eval = min(int(L_eval), L_train)

    t_eval = np.arange(L_eval, dtype=float)

    y_hat = generate_clean_data(t_eval, a_hat, omega_hat)
    y_ref = y_noisy[:L_eval]

    delta = y_hat - y_ref
    residual_norm = np.linalg.norm(delta, 2)

    norm_ref = np.linalg.norm(y_ref, 2)
    relative_residual = residual_norm / (norm_ref + eps)

    sqrt_L = np.sqrt(L_eval)
    rmse = residual_norm / sqrt_L
    rms_ref = norm_ref / sqrt_L
    rel_rmse = rmse / (rms_ref + eps)

    return residual_norm, relative_residual, rmse, rel_rmse
