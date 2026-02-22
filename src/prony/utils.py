import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["match_estimates", "relative_exponent_error", "wrap_to_nyquist"]


def match_estimates(
    a_true: np.ndarray,
    omega_true: np.ndarray,
    a_hat: np.ndarray,
    omega_hat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimal pairing of estimated components to ground truth using the
    Hungarian algorithm.

    Matching is performed in the pole (z) plane using z = exp(omega),
    which avoids phase wrapping issues in the exponent domain.

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
        Pole distances |z_true - z_hat| for each matched pair.

    Raises
    ------
    ValueError
        If any input array is not 1-D, if true and estimated arrays differ
        in length, or if a/omega pairs within the same set differ in length.

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
    a_true = np.asarray(a_true, dtype=complex)
    omega_true = np.asarray(omega_true, dtype=complex)
    a_hat = np.asarray(a_hat, dtype=complex)
    omega_hat = np.asarray(omega_hat, dtype=complex)

    if a_true.ndim != 1 or omega_true.ndim != 1:
        raise ValueError("a_true and omega_true must be 1-D arrays.")
    if a_hat.ndim != 1 or omega_hat.ndim != 1:
        raise ValueError("a_hat and omega_hat must be 1-D arrays.")
    if len(a_true) != len(omega_true):
        raise ValueError(
            f"a_true and omega_true must have the same length, "
            f"got {len(a_true)} and {len(omega_true)}."
        )
    if len(a_hat) != len(omega_hat):
        raise ValueError(
            f"a_hat and omega_hat must have the same length, "
            f"got {len(a_hat)} and {len(omega_hat)}."
        )
    if len(a_true) != len(a_hat):
        raise ValueError(
            f"True and estimated arrays must have the same length, "
            f"got {len(a_true)} true and {len(a_hat)} estimated components."
        )

    z_true = np.exp(omega_true)
    z_hat = np.exp(omega_hat)

    C = np.abs(z_true[:, None] - z_hat[None, :])
    row_ind, col_ind = linear_sum_assignment(C)

    return a_hat[col_ind], omega_hat[col_ind], C[row_ind, col_ind]


def relative_exponent_error(
    omega_true: np.ndarray,
    omega_hat_matched: np.ndarray
) -> float:
    """
    Compute a relative error between true and matched estimated exponents.

    For unit-circle exponents (real part ≈ 0), handles 2π wrapping by taking
    the minimal circular distance. For general complex exponents, uses direct
    Euclidean difference in the ω-plane, normalized by the norm of the true
    exponents.

    Parameters
    ----------
    omega_true : np.ndarray
        True complex exponents, shape (n,).
    omega_hat_matched : np.ndarray
        Estimated exponents after matching, same shape as omega_true.

    Returns
    -------
    float
        Relative exponent error (dimensionless, ≥ 0).

    Raises
    ------
    ValueError
        If inputs have mismatched shapes, are empty, or if true exponents
        have near-zero norm (making relative error undefined).

    Examples
    --------
    >>> omega_true = np.array([-0.1+0.5j, -0.2-0.3j])
    >>> omega_est = np.array([-0.11+0.49j, -0.19-0.31j])
    >>> err = relative_exponent_error(omega_true, omega_est)
    >>> err < 0.05
    True
    """
    omega_true = np.asarray(omega_true, dtype=complex)
    omega_hat_matched = np.asarray(omega_hat_matched, dtype=complex)

    if omega_true.shape != omega_hat_matched.shape:
        raise ValueError(
            f"omega_true and omega_hat_matched must have the same shape, "
            f"got {omega_true.shape} and {omega_hat_matched.shape}."
        )
    if omega_true.size == 0:
        raise ValueError("omega_true must not be empty.")

    # Detect unit-circle exponents relative to imaginary magnitude
    imag_scale = np.linalg.norm(np.imag(omega_true)) / len(omega_true)
    unit_circle = np.allclose(
        np.real(omega_true), 0.0, atol=max(1e-6, imag_scale * 1e-4)
    )

    if unit_circle:
        theta_t = np.mod(np.imag(omega_true), 2 * np.pi)
        theta_h = np.mod(np.imag(omega_hat_matched), 2 * np.pi)
        diff = np.abs(theta_t - theta_h)
        wrapped = np.minimum(diff, 2 * np.pi - diff)
        num = np.linalg.norm(wrapped)
        den = np.linalg.norm(theta_t)
        if den < np.finfo(float).eps:
            raise ValueError(
                "True exponents have near-zero norm — relative error is undefined."
            )
        return num / den
    else:
        num = np.linalg.norm(omega_true - omega_hat_matched)
        den = np.linalg.norm(omega_true)
        if den < np.finfo(float).eps:
            raise ValueError(
                "True exponents have near-zero norm — relative error is undefined."
            )
        return num / den


def wrap_to_nyquist(f: np.ndarray) -> np.ndarray:
    """
    Map real frequencies (in cycles per sample) into the Nyquist interval (-0.5, 0.5].

    Parameters
    ----------
    f : np.ndarray
        Real-valued frequencies in cycles/sample.

    Returns
    -------
    np.ndarray
        Frequencies wrapped to (-0.5, 0.5].

    Raises
    ------
    ValueError
        If f contains complex values.

    Examples
    --------
    >>> wrap_to_nyquist(np.array([0.7, -0.8, 0.3]))
    array([-0.3,  0.2,  0.3])
    """
    f = np.asarray(f)
    if np.iscomplexobj(f):
        raise ValueError(
            "wrap_to_nyquist expects real-valued frequencies, got complex input."
        )
    f = f.astype(float)
    return ((f + 0.5) % 1.0) - 0.5
