from __future__ import annotations

import numpy as np


def canonical_real_noise(seed: int, length: int) -> np.ndarray:
    """Accepted-abstract real Gaussian noise: default_rng(seed)."""
    rng = np.random.default_rng(int(seed))
    return rng.standard_normal(int(length))


def canonical_circular_noise(seed: int, length: int) -> np.ndarray:
    """Circular complex Gaussian noise with E|eps|^2 = 1."""
    rng = np.random.default_rng(int(seed))
    xi = rng.standard_normal(int(length))
    eta = rng.standard_normal(int(length))
    return (xi + 1j * eta) / np.sqrt(2.0)


def randomstate_real_noise(seed: int, length: int) -> np.ndarray:
    """Secondary replication using legacy RandomState."""
    rng = np.random.RandomState(int(seed))
    return rng.standard_normal(int(length))
