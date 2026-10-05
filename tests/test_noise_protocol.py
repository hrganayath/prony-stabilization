import numpy as np

from experiments.paired_oversampling.noise_protocol import (
    canonical_circular_noise,
    canonical_real_noise,
    randomstate_real_noise,
)


def test_canonical_real_noise_matches_default_rng():
    expected = np.random.default_rng(7).standard_normal(12)
    actual = canonical_real_noise(7, 12)
    np.testing.assert_allclose(actual, expected)


def test_canonical_circular_noise_is_reproducible_and_normalized():
    a = canonical_circular_noise(11, 100)
    b = canonical_circular_noise(11, 100)
    np.testing.assert_allclose(a, b)
    assert np.any(np.abs(np.imag(a)) > 0.0)


def test_randomstate_replication_is_distinct_from_canonical_stream():
    canonical = canonical_real_noise(0, 20)
    legacy = randomstate_real_noise(0, 20)
    assert not np.allclose(canonical, legacy)
