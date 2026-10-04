import numpy as np

from experiments.paired_oversampling.config import (
    AMPLITUDES,
    EXPONENTS,
    MAX_SAMPLES,
    MODEL_ORDER,
    SCORING_LENGTH,
    clean_signal,
    hankel_shape,
    sample_count,
)
from experiments.paired_oversampling.diagnostics import (
    _oracle_amplitudes,
    _subspace_diagnostics,
)
from prony import match_estimates, prony_method


def test_paired_config_dimensions():
    assert MODEL_ORDER == 4
    assert sample_count(1) == 2 * MODEL_ORDER + 1 == 9
    assert sample_count(10) == MAX_SAMPLES == 45
    assert hankel_shape(1) == (5, 5)
    assert hankel_shape(10) == (41, 5)
    assert SCORING_LENGTH == 9


def test_clean_signal_has_expected_length_and_is_finite():
    y = clean_signal(MAX_SAMPLES)
    assert y.shape == (MAX_SAMPLES,)
    assert np.all(np.isfinite(y))


def test_noiseless_fit_recovers_reference_signal():
    rho = 3
    n_samples = sample_count(rho)
    y = clean_signal(n_samples)

    a_hat, lambda_hat, _ = prony_method(
        y,
        oversampling_factor=rho,
        n=MODEL_ORDER,
    )
    a_hat, lambda_hat, _ = match_estimates(
        AMPLITUDES,
        EXPONENTS,
        a_hat,
        lambda_hat,
    )

    assert np.linalg.norm(lambda_hat - EXPONENTS) < 1e-10
    assert np.linalg.norm(a_hat - AMPLITUDES) < 1e-10


def test_noise_prefix_pairing_protocol():
    seed = 17
    rng = np.random.RandomState(seed)
    noise = rng.standard_normal(MAX_SAMPLES)

    n1 = sample_count(1)
    n4 = sample_count(4)

    assert np.array_equal(noise[:n1], noise[:n4][:n1])


def test_oracle_amplitudes_recover_noiseless_signal():
    rho = 5
    n_samples = sample_count(rho)
    y = clean_signal(n_samples)

    a_oracle = _oracle_amplitudes(y, n_samples)

    assert np.linalg.norm(a_oracle - AMPLITUDES) < 1e-10


def test_clean_subspace_angle_is_zero_for_identical_data():
    rho = 4
    n_samples = sample_count(rho)
    y = clean_signal(n_samples)

    angle_max, angle_mean = _subspace_diagnostics(
        y_clean=y,
        y_noisy=y,
        rho=rho,
    )

    assert angle_max < 1e-10
    assert angle_mean < 1e-10
