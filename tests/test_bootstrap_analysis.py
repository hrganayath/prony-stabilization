import numpy as np

from experiments.paired_oversampling.bootstrap_analysis import _paired_bootstrap_ci


def test_paired_bootstrap_detects_clear_positive_difference():
    rng = np.random.default_rng(0)
    x = np.array([2.0, 2.1, 1.9, 2.2, 2.05])
    y = np.array([1.0, 1.1, 0.9, 1.2, 1.05])
    diff, lo, hi, p = _paired_bootstrap_ci(
        x,
        y,
        n_bootstrap=2000,
        rng=rng,
    )
    assert diff > 0.0
    assert lo > 0.0
    assert hi > 0.0
    assert p == 1.0
