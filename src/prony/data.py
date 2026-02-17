import numpy as np
from typing import Union, List, Optional


def generate_clean_data(
    t: np.ndarray,
    a: Union[List[complex], np.ndarray],
    omegas: Union[List[complex], np.ndarray]
) -> np.ndarray:
    """
    Generate a clean sum-of-exponentials signal.

    The signal is defined as:
        y_clean(t) = Σ_i a[i] * exp(omegas[i] * t)

    Parameters
    ----------
    t : np.ndarray
        Time indices (real or complex, but typically real).
    a : list or np.ndarray of complex
        Complex amplitudes of each exponential component.
    omegas : list or np.ndarray of complex
        Complex exponents for each component.

    Returns
    -------
    y_clean : np.ndarray
        Signal values evaluated at each time in `t`, with shape (len(t),).

    Example
    -------
    >>> t = np.arange(5)
    >>> a = [1+0j, 0.5+0j]
    >>> omegas = [-0.1+0.5j, -0.2-0.3j]
    >>> y = generate_clean_data(t, a, omegas)
    >>> y.shape
    (5,)
    """
    a = np.asarray(a)
    omegas = np.asarray(omegas)

    y_clean = np.sum(a[:, None] * np.exp(omegas[:, None] * t), axis=0)
    return y_clean


def generate_noisy_data(
    y_clean: np.ndarray,
    noise_level: float,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Add Gaussian noise to a clean signal.

    Parameters
    ----------
    y_clean : np.ndarray
        Clean input signal.
    noise_level : float
        Standard deviation of the additive Gaussian noise.
    rng : numpy.random.Generator, optional
        Random number generator to use. If None, a new default generator is created.

    Returns
    -------
    y_noisy : np.ndarray
        Noisy signal, same shape as `y_clean`.

    Example
    -------
    >>> y_clean = np.array([1.0, 2.0, 3.0])
    >>> noise_level = 0.1
    >>> rng = np.random.default_rng(0)
    >>> y_noisy = generate_noisy_data(y_clean, noise_level, rng=rng)
    >>> y_noisy.shape
    (3,)
    """
    if rng is None:
        rng = np.random.default_rng()

    gaussian_noise = rng.normal(0.0, noise_level, size=y_clean.shape)
    return y_clean + gaussian_noise

