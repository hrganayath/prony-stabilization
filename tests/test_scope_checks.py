import numpy as np

from experiments.paired_oversampling.scope_checks import (
    MODELS,
    NOISE_LEVELS,
    RHO_VALUES,
    run,
)


def test_scope_model_dimensions():
    assert set(MODELS) == {6, 8}
    assert MODELS[6].sample_count(1) == 13
    assert MODELS[6].sample_count(10) == 67
    assert MODELS[8].sample_count(1) == 17
    assert MODELS[8].sample_count(10) == 89


def test_scope_models_are_purely_oscillatory():
    for model in MODELS.values():
        assert np.allclose(np.real(model.exponents), 0.0)
        assert np.all(np.isfinite(model.clean_signal(model.max_samples)))


def test_scope_check_smoke_shapes():
    results = run(n_trials=1)
    assert results[6].shape == (len(NOISE_LEVELS), len(RHO_VALUES), 1, 2)
    assert results[8].shape == (len(NOISE_LEVELS), len(RHO_VALUES), 1, 2)
    assert np.all(np.isfinite(results[6]))
    assert np.all(np.isfinite(results[8]))
