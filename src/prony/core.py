import numpy as np
from scipy.linalg import hankel, svd
from typing import Optional, Tuple

def prony_method(
    y: np.ndarray,
    oversampling_factor: int,
    n: int,
    rcond: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
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
        Oversampling factor ρ, determining the number of rows of the Hankel matrix
        as `ρ*n + 1`. Larger values improve noise robustness but increase
        computational cost.
    n : int
        Number of exponentials in the model.
    rcond : float, optional
        Cutoff for small singular values in the least squares solution for
        amplitudes. If None (default), machine precision is used.

    Returns
    -------
    a_est : np.ndarray
        Estimated complex amplitudes, shape (n,).
    omega_est : np.ndarray
        Estimated complex exponents, shape (n,). For a continuous-time model,
        the exponents correspond to `log(pole)` where `pole` are the eigenvalues
        of the shift matrix.
    cond_num : float
        Condition number of the Hankel matrix.

    Examples
    --------
    >>> t = np.arange(10)
    >>> a_true = np.array([1.0, 0.5])
    >>> omega_true = np.array([-0.1+0.5j, -0.2-0.3j])
    >>> y = np.sum([a_true[i]*np.exp(omega_true[i]*t) for i in range(2)], axis=0)
    >>> a_est, omega_est, cond = prony_method(y, oversampling_factor=2, n=2)
    >>> # Match estimates to true values (optional)
    >>> from prony.utils import match_estimates
    >>> a_est_m, omega_est_m, _ = match_estimates(a_true, omega_true, a_est, omega_est)
    >>> np.allclose(a_est_m, a_true, rtol=1e-2)
    True
    """
    # Number of rows in Hankel matrix (minus one)
    m = oversampling_factor * n

    # Construct Hankel matrix H of size (m+1) x (n+1)
    H = hankel(y[:(m+1)], y[m:m+(n+1)])

    # Condition number of the Hankel matrix
    cond_num = np.linalg.cond(H)

    # Singular value decomposition
    U, S, Vt = svd(H)

    # --- ESPRIT step ---
    # Signal subspace: first n left singular vectors
    U_s = U[:, :n]
    # Time-shifted subspaces
    U1, U2 = U_s[:-1, :], U_s[1:, :]
    # Solve for shift matrix Phi such that U2 ≈ U1 @ Phi
    Phi, _, _, _ = np.linalg.lstsq(U1, U2, rcond=None)
    # Poles are eigenvalues of Phi
    z_est, _ = np.linalg.eig(Phi)
    # Exponents (principal logarithm)
    omega_est = np.log(z_est.astype(complex))

    # --- Amplitude recovery ---
    L = m + n + 1                     # number of samples used in estimation
    # Vandermonde matrix with columns = exp(omega_est * t) for t = 0..L-1
    V = np.vander(np.exp(omega_est), L, increasing=True).T  # shape (L, n)

    # Column scaling to improve conditioning
    col_norms = np.linalg.norm(V, axis=0)
    col_norms[col_norms == 0] = 1.0
    V_scaled = V / col_norms

    # Set rcond for least squares if not provided
    if rcond is None:
        rcond = np.finfo(V_scaled.dtype).eps * max(V_scaled.shape)

    # Solve for scaled amplitudes
    a_scaled, _, _, _ = np.linalg.lstsq(V_scaled, y[:L], rcond=rcond)
    # Rescale
    a_est = a_scaled / col_norms

    return a_est, omega_est, cond_num
