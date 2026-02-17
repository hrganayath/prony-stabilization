# backward_analysis.py
import numpy as np
from typing import Optional, Tuple
from .data import generate_clean_data

def compute_backward_error(
    y_noisy: np.ndarray,
    a_hat: np.ndarray,
    omega_hat: np.ndarray,
    oversampling_factor: int,
    N: int,
    L_eval: Optional[int] = None,
    eps: float = 1e-15
) -> Tuple[float, float, float, float]:
    """
    Compute backward error and related metrics between estimated and noisy signals.

    The reconstruction uses the estimated parameters (a_hat, omega_hat) to generate
    a signal over the time indices 0..L_eval-1. The backward error is the ℓ2-norm
    of the difference between this reconstruction and the noisy data.

    Parameters
    ----------
    y_noisy : np.ndarray
        Noisy signal (must be at least L_eval long).
    a_hat : np.ndarray
        Estimated complex amplitudes.
    omega_hat : np.ndarray
        Estimated complex exponents.
    oversampling_factor : int
        Oversampling factor used in estimation (determines full window length).
    N : int
        Number of exponentials.
    L_eval : int, optional
        Number of samples to evaluate on. If None, uses the full window
        L_train = oversampling_factor * N + N + 1.
    eps : float, optional
        Small constant to avoid division by zero. Default 1e-15.

    Returns
    -------
    backward_error : float
        ℓ2 norm of the residual.
    relative_backward_error : float
        backward_error / (||y_noisy[:L_eval]||_2 + eps)
    rmse : float
        Root‑mean‑square error = backward_error / sqrt(L_eval)
    rel_rmse : float
        rmse / (RMS of reference signal + eps), where RMS is ||y_ref||_2 / sqrt(L_eval).

    Notes
    -----
    - When L_eval is None, the evaluation uses the full estimation window, which depends
      on oversampling_factor. This makes comparisons across different oversampling factors
      unfair because the window length varies. Use a fixed L_eval (e.g., 2*N+1) for
      cross‑factor comparisons.
    - The time vector is assumed to be integer indices starting at 0. This matches the
      construction in `generate_clean_data`.
    """
    # Full estimation window length
    L_train = oversampling_factor * N + N + 1
    # Determine evaluation length
    if L_eval is None:
        L_eval = L_train
    else:
        L_eval = min(int(L_eval), L_train)

    # Time indices (integer grid)
    t_eval = np.arange(L_eval, dtype=float)

    # Reconstruct signal on evaluation window
    y_hat = generate_clean_data(t_eval, a_hat, omega_hat)

    # Reference noisy segment
    y_ref = y_noisy[:L_eval]

    # Residual and norms
    delta = y_hat - y_ref
    backward_error = np.linalg.norm(delta, 2)
    norm_ref = np.linalg.norm(y_ref, 2)
    relative_backward_error = backward_error / (norm_ref + eps)

    # Per‑sample metrics
    sqrt_L = np.sqrt(L_eval)
    rmse = backward_error / sqrt_L
    rms_ref = norm_ref / sqrt_L
    rel_rmse = rmse / (rms_ref + eps)

    return backward_error, relative_backward_error, rmse, rel_rmse
