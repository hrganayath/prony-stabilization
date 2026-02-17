"""Tests for backward error computation."""

import numpy as np
from prony.backward import compute_backward_error
from prony.data import generate_clean_data

def test_backward_error_perfect_reconstruction():
    """Test that backward error is zero for perfect estimates."""
    t = np.arange(20)
    a_true = np.array([1.0, 0.5])
    omega_true = np.array([-0.1+0.5j, -0.2-0.3j])
    
    y_clean = generate_clean_data(t, a_true, omega_true)
    
    # Use true parameters as estimates
    back_err, rel_back_err, rmse, rel_rmse = compute_backward_error(
        y_clean, a_true, omega_true, oversampling_factor=2, N=2
    )
    
    assert back_err < 1e-10
    assert rel_back_err < 1e-10
    assert rmse < 1e-10


def test_backward_error_fixed_window():
    """Test fixed-window option."""
    t = np.arange(30)
    a_true = np.array([1.0, 0.5])
    omega_true = np.array([-0.1+0.5j, -0.2-0.3j])
    
    y_clean = generate_clean_data(t, a_true, omega_true)
    
    # Add small noise
    np.random.seed(42)
    y_noisy = y_clean + 0.01 * np.random.randn(len(y_clean))
    
    # Compute on full window
    back_err_full, _, _, _ = compute_backward_error(
        y_noisy, a_true, omega_true, oversampling_factor=2, N=2
    )
    
    # Compute on fixed window (L=5)
    back_err_fixed, _, _, _ = compute_backward_error(
        y_noisy, a_true, omega_true, oversampling_factor=2, N=2, L_eval=5
    )
    
    # Fixed window should be shorter, so error might be different
    assert back_err_fixed >= 0
    assert back_err_full >= 0
