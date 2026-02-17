# sensitivity_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from prony.data import generate_clean_data
from prony.core import prony_method
from prony.utils import match_estimates


def sensitivity_analysis(
    a_true: np.ndarray,
    omegas_true: np.ndarray,
    t: np.ndarray,
    os_factor: int,
    delta: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the Jacobian of the Prony estimates with respect to true parameters
    using central finite differences.

    For each true amplitude and exponent, a small relative perturbation (±δ) is applied,
    and the change in the estimated parameters is recorded. The estimates are first
    matched to the true components to ensure correct pairing.

    Parameters
    ----------
    a_true : np.ndarray
        True complex amplitudes, shape (N,).
    omegas_true : np.ndarray
        True complex exponents, shape (N,).
    t : np.ndarray
        Time vector (must be long enough for the chosen oversampling factor).
    os_factor : int
        Oversampling factor used in Prony's method.
    delta : float, optional
        Relative perturbation size (default 0.01). For amplitudes near zero, an absolute
        perturbation would be more appropriate; here we assume amplitudes are non‑zero.

    Returns
    -------
    J_a : np.ndarray
        Jacobian of amplitudes (N x N), complex. Entry (j,i) is ∂â_j / ∂a_true_i.
    J_omega : np.ndarray
        Jacobian of exponents (N x N), complex. Entry (j,i) is ∂ω̂_j / ∂ω_true_i.

    Notes
    -----
    The finite difference formula uses a central difference:
        ∂f/∂x ≈ (f(x+δx) - f(x-δx)) / (2δx)
    where δx = δ * x (relative perturbation). For x = 0 this would fail; the true
    parameters in this study are non‑zero, so it is safe.
    """
    N = len(a_true)
    J_a = np.zeros((N, N), dtype=complex)
    J_omega = np.zeros((N, N), dtype=complex)

    # Baseline estimates on clean signal
    y0 = generate_clean_data(t, a_true, omegas_true)
    a0, w0, _ = prony_method(y0, os_factor, n=N)

    # Match baseline to true to establish ordering
    a0_m, w0_m, _ = match_estimates(a_true, omegas_true, a0, w0)

    for i in range(N):
        for sign in (+1, -1):
            # ----- Perturb amplitude a_true[i] -----
            a_p = a_true.copy()
            # Relative perturbation (assumes a_true[i] != 0)
            a_p[i] *= (1 + sign * delta)
            y_p = generate_clean_data(t, a_p, omegas_true)
            a_est, w_est, _ = prony_method(y_p, os_factor, n=N)

            # Match perturbed estimates to true
            a_est_m, w_est_m, _ = match_estimates(a_true, omegas_true, a_est, w_est)

            # Finite difference for amplitude
            da = (a_est_m - a0_m) / (sign * delta * a_true[i])
            J_a[:, i] += da

            # ----- Perturb exponent omegas_true[i] -----
            w_p = omegas_true.copy()
            w_p[i] += sign * delta * omegas_true[i]   # relative perturbation
            y_p = generate_clean_data(t, a_true, w_p)
            a_est, w_est, _ = prony_method(y_p, os_factor, n=N)

            a_est_m, w_est_m, _ = match_estimates(a_true, omegas_true, a_est, w_est)

            dw = (w_est_m - w0_m) / (sign * delta * omegas_true[i])
            J_omega[:, i] += dw

        # Average over the two perturbations (central difference)
        J_a[:, i] /= 2
        J_omega[:, i] /= 2

    return J_a, J_omega


def plot_sensitivity(
    a_true: np.ndarray,
    omegas_true: np.ndarray,
    t: np.ndarray,
    os_factors: List[int],
    delta: float = 0.01,
    save_path: str = None
) -> None:
    """
    For each oversampling factor, compute the spectral norm of the Jacobian matrices
    and plot them as a function of oversampling.

    Parameters
    ----------
    a_true : np.ndarray
        True complex amplitudes.
    omegas_true : np.ndarray
        True complex exponents.
    t : np.ndarray
        Time vector.
    os_factors : list of int
        List of oversampling factors to evaluate.
    delta : float, optional
        Relative perturbation size (default 0.01).
    save_path : str, optional
        If provided, save the figure to this path.
    """
    amp_specs = []
    freq_specs = []

    for osf in os_factors:
        J_a, J_w = sensitivity_analysis(a_true, omegas_true, t, osf, delta)
        # Largest singular value = spectral norm
        sigma_a = np.linalg.svd(J_a, compute_uv=False)[0]
        sigma_w = np.linalg.svd(J_w, compute_uv=False)[0]
        amp_specs.append(sigma_a)
        freq_specs.append(sigma_w)

    plt.figure(figsize=(8, 5))
    plt.plot(os_factors, amp_specs, 'o-', label="Amplitude Sensitivity")
    plt.plot(os_factors, freq_specs, 's-', label="Frequency Sensitivity")
    plt.xlabel("Oversampling Factor")
    plt.ylabel("Spectral Norm of Jacobian")
    plt.title("Parameter Sensitivity vs. Oversampling")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
