"""Tests for signal generation functions in data.py."""
import numpy as np
import pytest
from prony.data import generate_clean_data, generate_noisy_data


# ── generate_clean_data ───────────────────────────────────────────────────────

def test_clean_data_output_shape():
    """Output shape must match length of t."""
    t = np.arange(10)
    a = np.array([1.0 + 0j, 0.5 + 0j])
    omegas = np.array([-0.1 + 0.5j, -0.2 - 0.3j])
    y = generate_clean_data(t, a, omegas)
    assert y.shape == (10,)


def test_clean_data_single_exponential():
    """Single exponential: y[k] = a * exp(omega * k)."""
    t = np.arange(5)
    a = np.array([2.0 + 0j])
    omega = np.array([0.0 + 0j])  # constant signal
    y = generate_clean_data(t, a, omega)
    np.testing.assert_allclose(y, np.full(5, 2.0 + 0j), atol=1e-12)


def test_clean_data_with_dt():
    """dt parameter scales t correctly."""
    t = np.arange(5)
    a = np.array([1.0 + 0j])
    omega = np.array([-0.1 + 0j])

    y_dt1 = generate_clean_data(t, a, omega, dt=1.0)
    y_dt2 = generate_clean_data(t, a, omega, dt=2.0)

    # With dt=2, effectively t is doubled, so decay is faster
    expected = np.exp(-0.1 * t * 2.0)
    np.testing.assert_allclose(np.real(y_dt2), expected, atol=1e-12)
    # dt=1 and dt=2 should give different results
    assert not np.allclose(y_dt1, y_dt2)


def test_clean_data_dtype_is_complex():
    """Output must always be complex even with real inputs."""
    t = np.arange(5)
    a = np.array([1.0, 0.5])        # real input
    omegas = np.array([-0.1, -0.2]) # real input
    y = generate_clean_data(t, a, omegas)
    assert np.iscomplexobj(y)


def test_clean_data_mismatched_a_omegas():
    """Mismatched a and omegas lengths should raise ValueError."""
    t = np.arange(5)
    a = np.array([1.0 + 0j, 0.5 + 0j])
    omegas = np.array([-0.1 + 0.5j])  # wrong length
    with pytest.raises(ValueError, match="same length"):
        generate_clean_data(t, a, omegas)


def test_clean_data_empty_t():
    """Empty t array should raise ValueError."""
    t = np.array([])
    a = np.array([1.0 + 0j])
    omegas = np.array([-0.1 + 0.5j])
    with pytest.raises(ValueError, match="empty"):
        generate_clean_data(t, a, omegas)


def test_clean_data_invalid_dt():
    """Non-positive dt should raise ValueError."""
    t = np.arange(5)
    a = np.array([1.0 + 0j])
    omegas = np.array([-0.1 + 0.5j])
    with pytest.raises(ValueError, match="dt must be positive"):
        generate_clean_data(t, a, omegas, dt=0.0)
    with pytest.raises(ValueError, match="dt must be positive"):
        generate_clean_data(t, a, omegas, dt=-1.0)


def test_clean_data_2d_a_raises():
    """2-D a array should raise ValueError."""
    t = np.arange(5)
    a = np.array([[1.0 + 0j, 0.5 + 0j]])  # 2-D
    omegas = np.array([-0.1 + 0.5j, -0.2 - 0.3j])
    with pytest.raises(ValueError, match="1-D"):
        generate_clean_data(t, a, omegas)


# ── generate_noisy_data ───────────────────────────────────────────────────────

def test_noisy_data_output_shape():
    """Output shape must match input y_clean."""
    y_clean = np.ones(10, dtype=complex)
    rng = np.random.default_rng(0)
    y_noisy = generate_noisy_data(y_clean, noise_level=0.1, rng=rng)
    assert y_noisy.shape == (10,)


def test_noisy_data_reproducible_with_seed():
    """Same rng seed must produce identical noisy signals."""
    y_clean = np.ones(10, dtype=complex)
    y1 = generate_noisy_data(y_clean, 0.1, rng=np.random.default_rng(42))
    y2 = generate_noisy_data(y_clean, 0.1, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(y1, y2)


def test_noisy_data_different_seeds_differ():
    """Different rng seeds should produce different outputs."""
    y_clean = np.ones(10, dtype=complex)
    y1 = generate_noisy_data(y_clean, 0.1, rng=np.random.default_rng(0))
    y2 = generate_noisy_data(y_clean, 0.1, rng=np.random.default_rng(1))
    assert not np.allclose(y1, y2)


def test_noisy_data_zero_noise_equals_clean():
    """noise_level=0 should return signal identical to y_clean."""
    y_clean = np.array([1.0 + 0.5j, 2.0 - 1.0j, 3.0 + 0j])
    rng = np.random.default_rng(0)
    y_noisy = generate_noisy_data(y_clean, noise_level=0.0, rng=rng)
    np.testing.assert_array_equal(y_noisy, y_clean)


def test_noisy_data_negative_noise_raises():
    """Negative noise_level should raise ValueError."""
    y_clean = np.ones(10, dtype=complex)
    with pytest.raises(ValueError, match="non-negative"):
        generate_noisy_data(y_clean, noise_level=-0.1)


def test_noisy_data_complex_noise_perturbs_both_parts():
    """complex_noise=True should perturb both real and imaginary parts."""
    rng = np.random.default_rng(0)
    y_clean = np.ones(100, dtype=complex) * (1.0 + 1.0j)
    y_noisy = generate_noisy_data(y_clean, noise_level=0.1,
                                   rng=rng, complex_noise=True)
    # Both parts should differ from the original
    assert not np.allclose(np.real(y_noisy), np.ones(100))
    assert not np.allclose(np.imag(y_noisy), np.ones(100))


def test_noisy_data_real_noise_only_perturbs_real_part():
    """complex_noise=False (default) should only add noise to real part."""
    rng = np.random.default_rng(0)
    y_clean = np.ones(100, dtype=complex) * (1.0 + 1.0j)
    y_noisy = generate_noisy_data(y_clean, noise_level=0.1,
                                   rng=rng, complex_noise=False)
    # Imaginary part should be unchanged
    np.testing.assert_array_equal(np.imag(y_noisy), np.ones(100))
    # Real part should be perturbed
    assert not np.allclose(np.real(y_noisy), np.ones(100))
