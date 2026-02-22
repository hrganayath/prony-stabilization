"""Tests for core Prony functionality."""
import numpy as np
import pytest
from prony.core import prony_method
from prony.utils import match_estimates, relative_exponent_error
from prony.data import generate_clean_data


def test_prony_method_basic():
    """Test Prony method on a simple 2-exponential signal."""
    t = np.arange(20)
    a_true = np.array([1.0, 0.5])
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j])

    y = generate_clean_data(t, a_true, omega_true)

    a_est, omega_est, cond = prony_method(y, oversampling_factor=2, n=2)

    a_est_m, omega_est_m, distances = match_estimates(
        a_true, omega_true, a_est, omega_est
    )

    assert np.allclose(a_est_m, a_true, rtol=1e-2, atol=1e-2)
    assert np.allclose(omega_est_m, omega_true, rtol=1e-2, atol=1e-2)
    assert cond >= 1.0, f"Condition number must be >= 1, got {cond}"
    assert np.isfinite(cond), f"Condition number must be finite, got {cond}"


def test_prony_method_unit_circle():
    """Test Prony method on unit-circle poles (no decay)."""
    t = np.arange(20)
    a_true = np.array([1.0, 1.0])
    f = np.array([0.1, 0.3])  # cycles/sample
    omega_true = 1j * 2 * np.pi * f

    y = generate_clean_data(t, a_true, omega_true)

    a_est, omega_est, cond = prony_method(y, oversampling_factor=2, n=2)
    a_est_m, omega_est_m, _ = match_estimates(a_true, omega_true, a_est, omega_est)

    freq_err = relative_exponent_error(omega_true, omega_est_m)
    assert freq_err < 1e-2


def test_prony_method_oversampling():
    """Test that higher oversampling improves conditioning."""
    t = np.arange(50)
    a_true = np.array([1.0, 0.5, 0.3])
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j, -0.15 + 0.8j])

    y = generate_clean_data(t, a_true, omega_true)

    _, _, cond_low = prony_method(y, oversampling_factor=2, n=3)
    _, _, cond_high = prony_method(y, oversampling_factor=4, n=3)

    # Higher oversampling should give better conditioning (lower condition number)
    assert cond_high < cond_low, (
        f"Expected higher oversampling to improve conditioning, "
        f"but cond_low={cond_low:.2e}, cond_high={cond_high:.2e}"
    )


def test_prony_method_single_exponential():
    """Test Prony method with a single exponential component."""
    t = np.arange(15)
    a_true = np.array([2.0 + 0j])
    omega_true = np.array([-0.05 + 0.3j])

    y = generate_clean_data(t, a_true, omega_true)

    a_est, omega_est, cond = prony_method(y, oversampling_factor=2, n=1)
    a_est_m, omega_est_m, _ = match_estimates(a_true, omega_true, a_est, omega_est)

    assert np.allclose(a_est_m, a_true, rtol=1e-2, atol=1e-2)
    assert np.allclose(omega_est_m, omega_true, rtol=1e-2, atol=1e-2)
    assert cond >= 1.0, f"Condition number must be >= 1, got {cond}"
    assert np.isfinite(cond), f"Condition number must be finite, got {cond}"


def test_prony_method_noisy_signal():
    """Test that Prony method produces reasonable estimates on a noisy signal."""
    rng = np.random.default_rng(42)
    t = np.arange(40)
    a_true = np.array([1.0, 0.5])
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j])

    y_clean = generate_clean_data(t, a_true, omega_true)
    y_noisy = y_clean + 0.01 * rng.standard_normal(len(t))

    a_est, omega_est, cond = prony_method(y_noisy, oversampling_factor=2, n=2)
    a_est_m, omega_est_m, _ = match_estimates(a_true, omega_true, a_est, omega_est)

    # Looser tolerances for noisy case
    assert np.allclose(a_est_m, a_true, rtol=0.1, atol=0.1)
    assert np.allclose(omega_est_m, omega_true, rtol=0.1, atol=0.1)
    assert cond >= 1.0
    assert np.isfinite(cond)


def test_prony_method_signal_too_short():
    """prony_method should raise ValueError when y is too short."""
    y = np.ones(5)
    with pytest.raises(ValueError, match="too short"):
        prony_method(y, oversampling_factor=2, n=3)


def test_prony_method_invalid_n():
    """prony_method should raise ValueError for n < 1."""
    y = np.ones(20)
    with pytest.raises(ValueError, match="n must be at least 1"):
        prony_method(y, oversampling_factor=2, n=0)


def test_prony_method_invalid_oversampling():
    """prony_method should raise ValueError for oversampling_factor < 1."""
    y = np.ones(20)
    with pytest.raises(ValueError, match="oversampling_factor must be at least 1"):
        prony_method(y, oversampling_factor=0, n=2)
