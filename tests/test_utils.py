"""Tests for utility functions (matching, error metrics, frequency wrapping)."""
import numpy as np
import pytest
from prony.utils import match_estimates, relative_exponent_error, wrap_to_nyquist


# ── match_estimates ───────────────────────────────────────────────────────────

def test_match_estimates_correct_reordering():
    """Matching should correctly reorder estimates to align with true components."""
    a_true = np.array([1.0, 0.5])
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j])

    # Estimates in wrong order
    a_est = np.array([0.48, 1.02])
    omega_est = np.array([-0.19 - 0.31j, -0.11 + 0.49j])

    a_matched, omega_matched, distances = match_estimates(
        a_true, omega_true, a_est, omega_est
    )

    # Second estimate should match first true component, and vice versa
    assert np.allclose(a_matched[0], 1.02, rtol=1e-2)
    assert np.allclose(a_matched[1], 0.48, rtol=1e-2)
    assert len(distances) == 2
    assert np.all(distances >= 0)


def test_match_estimates_already_ordered():
    """Matching an already-correct ordering should leave it unchanged."""
    a_true = np.array([1.0, 0.5])
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j])

    a_matched, omega_matched, distances = match_estimates(
        a_true, omega_true, a_true.copy(), omega_true.copy()
    )

    np.testing.assert_allclose(a_matched, a_true, atol=1e-12)
    np.testing.assert_allclose(omega_matched, omega_true, atol=1e-12)
    np.testing.assert_allclose(distances, np.zeros(2), atol=1e-12)


def test_match_estimates_distances_are_non_negative():
    """All matched pole distances must be non-negative."""
    a_true = np.array([1.0, 0.5, 0.3])
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j, -0.15 + 0.8j])
    a_est = np.array([0.31, 0.98, 0.49])
    omega_est = np.array([-0.14 + 0.81j, -0.09 + 0.51j, -0.21 - 0.29j])

    _, _, distances = match_estimates(a_true, omega_true, a_est, omega_est)

    assert np.all(distances >= 0)
    assert len(distances) == 3


def test_match_estimates_mismatched_a_omega_lengths():
    """match_estimates should raise ValueError when a and omega lengths differ."""
    a_true = np.array([1.0, 0.5])
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j])
    a_est = np.array([1.0])              # length 1
    omega_est = np.array([-0.1 + 0.5j]) # length 1 — but mismatches a_true

    with pytest.raises(ValueError, match="same length"):
        match_estimates(a_true, omega_true, a_est, omega_est)


def test_match_estimates_mismatched_true_vs_estimated():
    """match_estimates should raise ValueError when true and estimated sizes differ."""
    a_true = np.array([1.0, 0.5])
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j])
    a_est = np.array([1.0, 0.5, 0.3])
    omega_est = np.array([-0.1 + 0.5j, -0.2 - 0.3j, -0.15 + 0.8j])

    with pytest.raises(ValueError, match="same length"):
        match_estimates(a_true, omega_true, a_est, omega_est)


def test_match_estimates_non_1d_input_raises():
    """match_estimates should raise ValueError for non-1D inputs."""
    a_true = np.array([[1.0, 0.5]])   # 2-D
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j])
    a_est = np.array([1.0, 0.5])
    omega_est = np.array([-0.1 + 0.5j, -0.2 - 0.3j])

    with pytest.raises(ValueError, match="1-D"):
        match_estimates(a_true, omega_true, a_est, omega_est)


# ── relative_exponent_error ───────────────────────────────────────────────────

def test_relative_exponent_error_unit_circle_small_error():
    """Small perturbation of unit-circle exponents should give small error."""
    omega_true = np.array([1j * 0.5, 1j * 0.8])
    omega_est = np.array([1j * 0.51, 1j * 0.79])

    error = relative_exponent_error(omega_true, omega_est)
    assert error < 0.05


def test_relative_exponent_error_unit_circle_wrapping():
    """Unit-circle error should correctly handle 2π wrapping."""
    # 0.9 and -0.1 rad: circular distance = min(1.0, 2π - 1.0) = 1.0
    # Normalized by norm of true = 0.9, so error = 1.0 / 0.9 ≈ 1.111
    omega_true = np.array([1j * 0.9])
    omega_est = np.array([1j * -0.1])

    error = relative_exponent_error(omega_true, omega_est)
    expected = 1.0 / 0.9
    assert np.isclose(error, expected, rtol=1e-6), (
        f"Expected ~{expected:.4f}, got {error:.4f}"
    )


def test_relative_exponent_error_complex_small_error():
    """Small perturbation of general complex exponents should give small error."""
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j])
    omega_est = np.array([-0.11 + 0.49j, -0.19 - 0.31j])

    error = relative_exponent_error(omega_true, omega_est)
    assert error < 0.05


def test_relative_exponent_error_perfect_match():
    """Identical inputs should give zero error."""
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j])
    error = relative_exponent_error(omega_true, omega_true.copy())
    assert np.isclose(error, 0.0, atol=1e-12)


def test_relative_exponent_error_mismatched_shapes():
    """relative_exponent_error should raise ValueError for mismatched shapes."""
    omega_true = np.array([-0.1 + 0.5j, -0.2 - 0.3j])
    omega_est = np.array([-0.1 + 0.5j])  # wrong length

    with pytest.raises(ValueError, match="same shape"):
        relative_exponent_error(omega_true, omega_est)


def test_relative_exponent_error_empty_input():
    """relative_exponent_error should raise ValueError for empty input."""
    with pytest.raises(ValueError, match="empty"):
        relative_exponent_error(np.array([]), np.array([]))


def test_relative_exponent_error_zero_norm():
    """relative_exponent_error should raise ValueError when true norm is zero."""
    omega_true = np.array([0.0 + 0.0j, 0.0 + 0.0j])
    omega_est = np.array([0.01 + 0.0j, 0.0 + 0.01j])

    with pytest.raises(ValueError, match="near-zero norm"):
        relative_exponent_error(omega_true, omega_est)


# ── wrap_to_nyquist ───────────────────────────────────────────────────────────

def test_wrap_to_nyquist_basic():
    """Test frequency wrapping to (-0.5, 0.5]."""
    f = np.array([0.7, -0.8, 1.2, 0.0])
    f_wrapped = wrap_to_nyquist(f)

    expected = np.array([-0.3, 0.2, 0.2, 0.0])
    np.testing.assert_allclose(f_wrapped, expected, atol=1e-12)


def test_wrap_to_nyquist_distinct_outputs():
    """Values mapping to distinct outputs should all be correct."""
    f = np.array([0.7, -0.8, 0.3])
    f_wrapped = wrap_to_nyquist(f)

    expected = np.array([-0.3, 0.2, 0.3])
    np.testing.assert_allclose(f_wrapped, expected, atol=1e-12)


def test_wrap_to_nyquist_already_in_range():
    """Values already in (-0.5, 0.5] should be unchanged."""
    f = np.array([-0.4, 0.0, 0.5, -0.5 + 1e-10])
    f_wrapped = wrap_to_nyquist(f)
    np.testing.assert_allclose(f_wrapped, f, atol=1e-9)


def test_wrap_to_nyquist_output_shape():
    """Output shape must match input shape."""
    f = np.array([0.1, 0.6, -0.7, 1.1, -1.2])
    assert wrap_to_nyquist(f).shape == f.shape


def test_wrap_to_nyquist_complex_input_raises():
    """wrap_to_nyquist should raise ValueError for complex input."""
    with pytest.raises(ValueError, match="real-valued"):
        wrap_to_nyquist(np.array([0.1 + 0.2j, 0.3 + 0.4j]))
