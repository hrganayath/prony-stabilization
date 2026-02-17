import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Tuple

def match_estimates(
    a_true: np.ndarray,
    omega_true: np.ndarray,
    a_hat: np.ndarray,
    omega_hat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimal pairing of estimated components to ground truth using Hungarian algorithm.

    The matching is performed in the pole (z) plane, i.e., using z = exp(omega),
    which is universally valid and avoids phase wrapping issues.

    Parameters
    ----------
    a_true : np.ndarray
        True complex amplitudes, shape (n,).
    omega_true : np.ndarray
        True complex exponents, shape (n,).
    a_hat : np.ndarray
        Estimated complex amplitudes, shape (n,).
    omega_hat : np.ndarray
        Estimated complex exponents, shape (n,).

    Returns
    -------
    a_hat_matched : np.ndarray
        Estimated amplitudes reordered to match the order of true components.
    omega_hat_matched : np.ndarray
        Estimated exponents reordered accordingly.
    distances : np.ndarray
        Pole distances (|z_true - z_hat|) for the matched pairs.

    Examples
    --------
    >>> a_true = np.array([1.0, 0.5])
    >>> omega_true = np.array([-0.1+0.5j, -0.2-0.3j])
    >>> a_est = np.array([0.48, 1.02])
    >>> omega_est = np.array([-0.19-0.31j, -0.11+0.49j])
    >>> a_m, w_m, d = match_estimates(a_true, omega_true, a_est, omega_est)
    >>> a_m
    array([1.02+0.j, 0.48+0.j])
    """
    a_true = np.asarray(a_true)
    omega_true = np.asarray(omega_true)
    a_hat = np.asarray(a_hat)
    omega_hat = np.asarray(omega_hat)

    # Poles in z-plane
    z_true = np.exp(omega_true)
    z_hat = np.exp(omega_hat)

    # Cost matrix: absolute differences between poles
    C = np.abs(z_true[:, None] - z_hat[None, :])
    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(C)

    return a_hat[col_ind], omega_hat[col_ind], C[row_ind, col_ind]


def frequency_error_generic(
    omega_true: np.ndarray,
    omega_hat_matched: np.ndarray
) -> float:
    """
    Compute a relative frequency error between true and matched estimated exponents.

    For unit‑circle exponents (real part ≈ 0), the error handles 2π wrapping by
    taking the minimal circular distance. For general complex exponents, a direct
    Euclidean difference in the ω‑plane is used, normalized by the norm of the true
    exponents.

    Parameters
    ----------
    omega_true : np.ndarray
        True complex exponents, shape (n,).
    omega_hat_matched : np.ndarray
        Estimated exponents after matching, same shape.

    Returns
    -------
    float
        Relative frequency error.

    Examples
    --------
    >>> omega_true = np.array([-0.1+0.5j, -0.2-0.3j])
    >>> omega_est = np.array([-0.11+0.49j, -0.19-0.31j])
    >>> freq_err = frequency_error_generic(omega_true, omega_est)
    >>> freq_err < 0.05
    True
    """
    omega_true = np.asarray(omega_true)
    omega_hat_matched = np.asarray(omega_hat_matched)

    # Check if true exponents lie on the unit circle (real part ~ 0)
    unit_circle = np.allclose(np.real(omega_true), 0.0, atol=1e-12)

    if unit_circle:
        # Extract imaginary parts (frequencies) and wrap to [0, 2π)
        theta_t = np.mod(np.imag(omega_true), 2 * np.pi)
        theta_h = np.mod(np.imag(omega_hat_matched), 2 * np.pi)
        diff = np.abs(theta_t - theta_h)
        # Circular distance (min of diff and 2π - diff)
        wrapped = np.minimum(diff, 2 * np.pi - diff)
        num = np.linalg.norm(wrapped)
        # Denominator: norm of the true frequencies (no need for np.abs, theta_t is real)
        den = np.linalg.norm(theta_t) + 1e-15
        return num / den
    else:
        # General case: direct Euclidean difference
        num = np.linalg.norm(omega_true - omega_hat_matched)
        den = np.linalg.norm(omega_true) + 1e-15
        return num / den


def wrap_to_nyquist(f: np.ndarray) -> np.ndarray:
    """
    Map real frequencies (in cycles per sample) into the Nyquist interval (-0.5, 0.5].

    This function is provided for convenience when displaying frequencies, but
    is not used in the core estimation or error computation.

    Parameters
    ----------
    f : np.ndarray
        Frequencies in cycles/sample (real).

    Returns
    -------
    np.ndarray
        Frequencies wrapped to (-0.5, 0.5].

    Examples
    --------
    >>> wrap_to_nyquist(np.array([0.7, -0.8, 1.2]))
    array([-0.3,  0.2,  0.2])
    """
    f = np.asarray(f, dtype=float)
    return ((f + 0.5) % 1.0) - 0.5
