"""Tests for residual (data misfit) computation."""
import numpy as np
import pytest
from prony.residual import compute_residual_error, ResidualMetrics
from prony.data import generate_clean_data, generate_noisy_data


def test_residual_error_perfect_reconstruction():
    """Perfect reconstruction should yield near-zero residuals."""
    t = np.arange(20)
    a_true = np.array([1.0, 0.5])
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j])
    y_clean = generate_clean_data(t, a_true, omega_true)

    metrics = compute_residual_error(
        y_clean, a_true, omega_true, oversampling_factor=2, n=2
    )

    assert metrics.residual_norm < 1e-10
    assert metrics.relative_residual < 1e-10
    assert metrics.rmse < 1e-10


def test_residual_error_fixed_window():
    """Smaller L_eval window should give residual <= full window residual."""
    t = np.arange(30)
    a_true = np.array([1.0, 0.5])
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j])
    y_clean = generate_clean_data(t, a_true, omega_true)

    rng = np.random.default_rng(42)
    y_noisy = generate_noisy_data(y_clean, noise_level=0.01, rng=rng)

    metrics_full = compute_residual_error(
        y_noisy, a_true, omega_true, oversampling_factor=2, n=2
    )
    metrics_fixed = compute_residual_error(
        y_noisy, a_true, omega_true, oversampling_factor=2, n=2, L_eval=5
    )

    assert metrics_fixed.residual_norm >= 0
    assert metrics_full.residual_norm >= 0
    # Smaller window should yield a smaller or equal residual norm
    assert metrics_fixed.residual_norm <= metrics_full.residual_norm + 1e-10, (
        f"Expected metrics_fixed.residual_norm={metrics_fixed.residual_norm:.4e} "
        f"<= metrics_full.residual_norm={metrics_full.residual_norm:.4e}"
    )


def test_residual_error_named_tuple_access():
    """ResidualMetrics fields should be accessible by name."""
    t = np.arange(20)
    a_true = np.array([1.0, 0.5])
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j])
    y_clean = generate_clean_data(t, a_true, omega_true)

    metrics = compute_residual_error(
        y_clean, a_true, omega_true, oversampling_factor=2, n=2
    )

    assert isinstance(metrics, ResidualMetrics)
    assert hasattr(metrics, "residual_norm")
    assert hasattr(metrics, "relative_residual")
    assert hasattr(metrics, "rmse")
    assert hasattr(metrics, "rel_rmse")
    assert metrics.residual_norm < 1e-10


def test_residual_error_all_metrics_non_negative():
    """All returned metrics must be non-negative."""
    t = np.arange(30)
    a_true = np.array([1.0, 0.5])
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j])
    y_clean = generate_clean_data(t, a_true, omega_true)

    rng = np.random.default_rng(0)
    y_noisy = generate_noisy_data(y_clean, noise_level=0.05, rng=rng)

    metrics = compute_residual_error(
        y_noisy, a_true, omega_true, oversampling_factor=2, n=2
    )

    assert metrics.residual_norm >= 0
    assert metrics.relative_residual >= 0
    assert metrics.rmse >= 0
    assert metrics.rel_rmse >= 0


def test_residual_error_signal_too_short():
    """compute_residual_error should raise ValueError when y_noisy is too short."""
    y = np.ones(5, dtype=complex)
    a_hat = np.array([1.0 + 0j])
    omega_hat = np.array([-0.1 + 0.5j])

    with pytest.raises(ValueError, match="too short"):
        compute_residual_error(y, a_hat, omega_hat, oversampling_factor=2, n=1)


def test_residual_error_invalid_n():
    """compute_residual_error should raise ValueError for n < 1."""
    y = np.ones(20, dtype=complex)
    a_hat = np.array([1.0 + 0j])
    omega_hat = np.array([-0.1 + 0.5j])

    with pytest.raises(ValueError, match="n must be at least 1"):
        compute_residual_error(y, a_hat, omega_hat, oversampling_factor=2, n=0)


def test_residual_error_invalid_L_eval():
    """compute_residual_error should raise ValueError when L_eval > len(y_noisy)."""
    y = np.ones(20, dtype=complex)
    a_hat = np.array([1.0 + 0j])
    omega_hat = np.array([-0.1 + 0.5j])

    with pytest.raises(ValueError, match="exceeds"):
        compute_residual_error(
            y, a_hat, omega_hat, oversampling_factor=2, n=1, L_eval=999
        )


def test_residual_error_mismatched_a_omega():
    """compute_residual_error should raise ValueError when a_hat and omega_hat lengths differ."""
    t = np.arange(20)
    a_true = np.array([1.0, 0.5])
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j])
    y_clean = generate_clean_data(t, a_true, omega_true)

    a_hat = np.array([1.0 + 0j])           # length 1
    omega_hat = np.array([-0.1 + 0.5j, -0.2 - 0.3j])  # length 2

    with pytest.raises(ValueError, match="same length"):
        compute_residual_error(
            y_clean, a_hat, omega_hat, oversampling_factor=2, n=2
        )
