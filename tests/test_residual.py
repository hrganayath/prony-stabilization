"""Tests for residual (data misfit) computation."""

import numpy as np
from prony.residual import compute_residual_error
from prony.data import generate_clean_data


def test_residual_error_perfect_reconstruction():
    t = np.arange(20)
    a_true = np.array([1.0, 0.5])
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j])

    y_clean = generate_clean_data(t, a_true, omega_true)

    res, rel_res, rmse, _ = compute_residual_error(
        y_clean, a_true, omega_true, oversampling_factor=2, N=2
    )

    assert res < 1e-10
    assert rel_res < 1e-10
    assert rmse < 1e-10


def test_residual_error_fixed_window():
    t = np.arange(30)
    a_true = np.array([1.0, 0.5])
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j])

    y_clean = generate_clean_data(t, a_true, omega_true)
    rng = np.random.default_rng(42)
    y_noisy = y_clean + 0.01 * rng.standard_normal(len(y_clean))

    res_full, _, _, _ = compute_residual_error(
        y_noisy, a_true, omega_true, oversampling_factor=2, N=2
    )
    res_fixed, _, _, _ = compute_residual_error(
        y_noisy, a_true, omega_true, oversampling_factor=2, N=2, L_eval=5
    )

    assert res_fixed >= 0
    assert res_full >= 0
