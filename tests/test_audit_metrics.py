import numpy as np

from experiments.paired_oversampling.audit_metrics import (
    _branch_jump_count,
    _draw_noise,
    run,
)
from experiments.paired_oversampling.config import MAX_SAMPLES


def test_real_audit_noise_has_zero_imaginary_part():
    noise = _draw_noise(7, "real")
    assert noise.shape == (MAX_SAMPLES,)
    assert np.allclose(np.imag(noise), 0.0)


def test_circular_audit_noise_is_reproducible():
    noise1 = _draw_noise(7, "circular")
    noise2 = _draw_noise(7, "circular")
    np.testing.assert_allclose(noise1, noise2)
    assert np.any(np.abs(np.imag(noise1)) > 0.0)


def test_branch_jump_count_detects_raw_two_pi_crossing():
    omega_true = np.array([-0.1 - 2.9j])
    omega_hat = np.array([-0.1 + (2.0 * np.pi - 2.9) * 1j])
    assert _branch_jump_count(omega_true, omega_hat) == 1


def test_audit_runner_smoke_shapes():
    raw, amplitudes, exponents = run(noise_model="real", n_trials=1)
    assert raw.shape == (10, 10, 1, 6)
    assert amplitudes.shape == (10, 10, 1, 4)
    assert exponents.shape == (10, 10, 1, 4)
    assert np.all(np.isfinite(raw))
