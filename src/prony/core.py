import warnings
import numpy as np
from scipy.linalg import hankel, svd

__all__ = ["prony_method"]


def prony_method(
    y: np.ndarray,
    oversampling_factor: int,
    n: int,
    rcond: float | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate amplitudes and exponents using Prony's method with SVD-based ESPRIT.

    The method constructs a Hankel matrix from the signal, computes its SVD,
    then uses ESPRIT to estimate the poles. Amplitudes are recovered via
    least squares on a Vandermonde system, with column scaling for improved
    conditioning.

    Parameters
    ----------
    y : np.ndarray
        Input signal, length at least `oversampling_factor * n + n + 1`.
    oversampling_factor : int
        Oversampling factor ρ ≥ 1. Determines the number of rows of the Hankel
        matrix as `ρ*n + 1`. Larger values improve noise robustness but increase
        computational cost.
    n : int
        Number of exponentials in the model (must be ≥ 1).
    rcond : float, optional
        Cutoff for small singular values in the least squares solution for
        amplitudes. If None (default), NumPy's optimal default is used.

    Returns
    -------
    a_est : np.ndarray
        Estimated complex amplitudes, shape (n,).
    omega_est : np.ndarray
        Estimated complex exponents, shape (n,), sorted by imaginary part.
        These are principal logarithms of the estimated poles.
    cond_num : float
        Condition number of the Hankel matrix.

    Raises
    ------
    ValueError
        If n < 1, oversampling_factor < 1, y is too short, or any estimated
        pole is near zero.

    Examples
    --------
    >>> import numpy as np
    >>> from prony.data import generate_clean_data
    >>> t = np.arange(10)
    >>> a_true = np.array([1.0, 0.5])
    >>> omega_true = np.array([-0.1+0.5j, -0.2-0.3j])
    >>> y = generate_clean_data(t, a_true, omega_true)
    >>> a_est, omega_est, cond = prony_method(y, oversampling_factor=2, n=2)
    """
    # --- Input validation ---
    if n < 1:
        raise ValueError(f"n must be at least 1, got n={n}.")
    if oversampling_factor < 1:
        raise ValueError(
            f"oversampling_factor must be at least 1, got {oversampling_factor}."
        )
    min_length = oversampling_factor * n + n + 1
    if len(y) < min_length:
        raise ValueError(
            f"Signal y is too short. Need at least {min_length} samples "
            f"for oversampling_factor={oversampling_factor} and n={n}, "
            f"but got len(y)={len(y)}."
        )

    # --- Hankel matrix ---
    m = oversampling_factor * n
    # Shape: (oversampling_factor*n + 1) x (n+1)
    H = hankel(y[:(m + 1)], y[m:m + (n + 1)])
    cond_num = np.linalg.cond(H)

    # --- SVD ---
    U, S, _ = svd(H, full_matrices=True)

    # Warn if there is no clear gap between signal and noise subspace
    if len(S) > n:
        gap = S[n - 1] / S[n]
        if gap < 10.0:
            warnings.warn(
                f"Weak singular value gap at rank n={n}: "
                f"S[n-1]/S[n]={gap:.2f}. The chosen n may not match the "
                f"signal's true rank. Consider inspecting the singular value "
                f"spectrum.",
                UserWarning,
                stacklevel=2,
            )

    # --- ESPRIT ---
    # Signal subspace: first n left singular vectors
    U_s = U[:, :n]
    U1, U2 = U_s[:-1, :], U_s[1:, :]

    # Shift matrix Phi such that U2 ≈ U1 @ Phi
    Phi, _, _, _ = np.linalg.lstsq(U1, U2, rcond=None)

    # Poles: eigenvalues of Phi, sorted by imaginary part of log
    z_est, _ = np.linalg.eig(Phi)
    z_est = z_est.astype(complex)
    sort_idx = np.argsort(np.imag(np.log(z_est)))
    z_est = z_est[sort_idx]

    # Guard against poles near zero before taking log
    near_zero = np.abs(z_est) < np.finfo(float).eps * 100
    if np.any(near_zero):
        raise ValueError(
            f"One or more estimated poles are near zero (|z| < machine "
            f"epsilon). This usually indicates severe noise or incorrect n. "
            f"Pole magnitudes: {np.abs(z_est)}"
        )

    omega_est = np.log(z_est)

    # --- Amplitude recovery ---
    # L: total sample window = number of unique Hankel entries
    L = m + n + 1

    # Vandermonde: V[k, i] = exp(omega_est[i] * k), shape (L, n)
    t_idx = np.arange(L)
    V = np.exp(np.outer(t_idx, omega_est))

    # Column scaling to improve conditioning
    col_norms = np.linalg.norm(V, axis=0)
    threshold = np.finfo(float).eps * np.max(col_norms)
    col_norms[col_norms < threshold] = 1.0
    V_scaled = V / col_norms

    # Fit amplitudes over the same window used to construct the Hankel matrix
    a_scaled, _, _, _ = np.linalg.lstsq(V_scaled, y[:L], rcond=rcond)
    a_est = a_scaled / col_norms

    return a_est, omega_est, cond_num
