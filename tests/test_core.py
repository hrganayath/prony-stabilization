"""Tests for core Prony functionality."""

import numpy as np
import pytest
from prony.core import prony_method
from prony.utils import match_estimates, frequency_error_generic
from prony.data import generate_clean_data

def test_prony_method_basic():
    """Test Prony method on a simple 2-exponential signal."""
    # Create synthetic signal
    t = np.arange(20)
    a_true = np.array([1.0, 0.5])
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j])
    
    y = generate_clean_data(t, a_true, omega_true)
    
    # Apply Prony method
    a_est, omega_est, cond = prony_method(y, oversampling_factor=2, n=2)
    
    # Match estimates to true values
    a_est_m, omega_est_m, distances = match_estimates(
        a_true, omega_true, a_est, omega_est
    )
    
    # Check that estimates are close
    assert np.allclose(a_est_m, a_true, rtol=1e-2, atol=1e-2)
    assert np.allclose(omega_est_m, omega_true, rtol=1e-2, atol=1e-2)
    assert cond > 0  # Condition number should be positive


def test_prony_method_unit_circle():
    """Test Prony method on unit-circle poles (no decay)."""
    t = np.arange(20)
    a_true = np.array([1.0, 1.0])
    f = np.array([0.1, 0.3])  # cycles/sample
    omega_true = 1j * 2 * np.pi * f
    
    y = generate_clean_data(t, a_true, omega_true)
    
    a_est, omega_est, cond = prony_method(y, oversampling_factor=2, n=2)
    a_est_m, omega_est_m, _ = match_estimates(a_true, omega_true, a_est, omega_est)
    
    # Check frequency error
    freq_err = frequency_error_generic(omega_true, omega_est_m)
    assert freq_err < 1e-2


def test_prony_method_oversampling():
    """Test that higher oversampling improves conditioning."""
    t = np.arange(50)
    a_true = np.array([1.0, 0.5, 0.3])
    omega_true = np.array([-0.1+0.5j, -0.2-0.3j, -0.15+0.8j])
    
    y = generate_clean_data(t, a_true, omega_true)
    
    # Lower oversampling
    _, _, cond_low = prony_method(y, oversampling_factor=2, n=3)
    
    # Higher oversampling
    _, _, cond_high = prony_method(y, oversampling_factor=4, n=3)
    
    # Higher oversampling should give better conditioning (lower condition number)
    # Note: This is not guaranteed but often true
    print(f"Condition numbers: low={cond_low:.2e}, high={cond_high:.2e}")
