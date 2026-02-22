import warnings
import numpy as np
from typing import NamedTuple
from .data import generate_clean_data

__all__ = ["ResidualMetrics", "compute_residual_error"]


class ResidualMetrics(NamedTuple):
    """Named tuple holding the four residual metrics from compute_residual_error."""
    residual_norm: float
    relative_residual: float
    rmse: float
    rel_rmse: float


def compute_residual_error(
    y_noisy: np.ndarray,
    a_hat: np.ndarray,
    omega_hat: np.ndarray,
    oversampling_factor: int,
    n: int,
    L_eval: int | None = None,
    eps: float = np.finfo(float).eps,
) -> ResidualMetrics:
    """
    Compute residual (data misfit) metrics between a reconstructed signal
    and the observed noisy samples.

    This is NOT a backward error in the numerical-analysis sense. It measures
    the ℓ2 data misfit:
        residual = || y_hat - y_ref ||_2
    where y_hat is reconstructed from (a_hat, omega_hat) and y_ref is the
    observed data over the evaluation window.

    Parameters
    ----------
    y_noisy : np.ndarray
        Observed (noisy) signal, 1-D, length >= L_eval.
    a_hat : np.ndarray
        Estimated complex amplitudes, shape (n,).
    omega_hat : np.ndarray
        Estimated complex exponents, shape (n,).
    oversampling_factor : int
        Oversampling factor used in prony_method. Determines the training
        window: L_train = oversampling_factor * n + n + 1.
    n : int
        Number of exponentials in the model (must be >= 1).
    L_eval : int, optional
        Number of samples to evaluate the residual over. If None (default),
        uses the full training window L_train. Must not exceed len(y_noisy).
    eps : float, optional
        Small value added to denominators to prevent division by zero.
        Defaults to machine epsilon (np.finfo(float).eps).

    Returns
    -------
    ResidualMetrics
        A named tuple with fields:
        - residual_norm : ||y_hat - y_ref||_2
        - relative_residual : residual_norm / ||y_ref||_2
        - rmse : residual_norm / sqrt(L_eval)
        - rel_rmse : rmse / rms(y_ref)

    Raises
    ------
    ValueError
        If inputs are invalid (wrong shape, too short, non-positive n, etc.)

    Examples
    --------
    >>> import numpy as np
    >>> from prony.data import generate_clean_data, generate_noisy_data
    >>> from prony.core import prony_method
    >>> t = np.arange(30)
    >>> a_true = np.array([1.0, 0.5])
    >>> omega_true = np.array([-0.1+0.5j, -0.2-0.3j])
    >>> y_clean = generate_clean_data(t, a_true, omega_true)
    >>> rng = np.random.default_rng(0)
    >>> y_noisy = generate_noisy_data(y_clean, noise_level=0.01, rng=rng)
    >>> a_hat, omega_hat, _ = prony_method(y_noisy, oversampling_factor=2, n=2)
    >>> metrics = compute_residual_error(
    ...     y_noisy, a_hat, omega_hat, oversampling_factor=2, n=2
    ... )
    >>> metrics.rel_rmse < 0.1
    True
    """
    # --- Input validation ---
    y_noisy = np.asarray(y_noisy)
    a_hat = np.asarray(a_hat, dtype=complex)
    omega_hat = np.asarray(omega_hat, dtype=complex)

    if y_noisy.ndim != 1:
        raise ValueError(f"y_noisy must be 1-D, got shape {y_noisy.shape}.")
    if a_hat.ndim != 1 or omega_hat.ndim != 1:
        raise ValueError("a_hat and omega_hat must be 1-D arrays.")
    if len(a_hat) != len(omega_hat):
        raise ValueError(
            f"a_hat and omega_hat must have the same length, "
            f"got {len(a_hat)} and {len(omega_hat)}."
        )
    if n < 1:
        raise ValueError(f"n must be at least 1, got n={n}.")
    if oversampling_factor < 1:
        raise ValueError(
            f"oversampling_factor must be at least 1, got {oversampling_factor}."
        )
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got eps={eps}.")

    L_train = oversampling_factor * n + n + 1

    if len(y_noisy) < L_train:
        raise ValueError(
            f"y_noisy is too short. Need at least {L_train} samples "
            f"for oversampling_factor={oversampling_factor} and n={n}, "
            f"but got len(y_noisy)={len(y_noisy)}."
        )

    if L_eval is None:
        L_eval = L_train
    else:
        L_eval = int(L_eval)
        if L_eval < 1:
            raise ValueError(f"L_eval must be at least 1, got L_eval={L_eval}.")
        if L_eval > len(y_noisy):
            raise ValueError(
                f"L_eval={L_eval} exceeds the length of y_noisy={len(y_noisy)}."
            )

    # --- Reconstruction and residual ---
    t_eval = np.arange(L_eval, dtype=float)
    y_hat = generate_clean_data(t_eval, a_hat, omega_hat)
    y_ref = y_noisy[:L_eval]

    delta = y_hat - y_ref
    residual_norm = np.linalg.norm(delta)       # L2 norm (default for 1-D)
    norm_ref = np.linalg.norm(y_ref)            # L2 norm (default for 1-D)

    if norm_ref < eps:
        warnings.warn(
            "Reference signal y_ref has near-zero norm. "
            "Relative residual metrics are unreliable.",
            UserWarning,
            stacklevel=2,
        )

    relative_residual = residual_norm / (norm_ref + eps)

    sqrt_L = np.sqrt(L_eval)
    rmse = residual_norm / sqrt_L
    rms_ref = norm_ref / sqrt_L

    if rms_ref < eps:
        warnings.warn(
            "Reference signal y_ref has near-zero RMS. "
            "rel_rmse is unreliable.",
            UserWarning,
            stacklevel=2,
        )

    rel_rmse = rmse / (rms_ref + eps)

    return ResidualMetrics(residual_norm, relative_residual, rmse, rel_rmse)
