import numpy as np

__all__ = ["generate_clean_data", "generate_noisy_data"]


def generate_clean_data(
    t: np.ndarray,
    a: list[complex] | np.ndarray,
    omegas: list[complex] | np.ndarray,
    dt: float = 1.0
) -> np.ndarray:
    """
    Generate a clean sum-of-exponentials signal.

    The signal is defined as:
        y_clean[k] = Σ_i a[i] * exp(omegas[i] * t[k] * dt)

    Parameters
    ----------
    t : np.ndarray
        Sample indices or time values (real-valued, 1-D).
    a : list or np.ndarray of complex
        Complex amplitudes of each exponential component.
    omegas : list or np.ndarray of complex
        Complex exponents for each component.
    dt : float, optional
        Sampling interval. Default is 1.0 (t treated as dimensionless indices).
        Set to the physical time step when t contains physical time values.

    Returns
    -------
    y_clean : np.ndarray
        Complex signal values at each time in t, shape (len(t),).

    Raises
    ------
    ValueError
        If a and omegas have different lengths, t is empty, or dt <= 0.

    Example
    -------
    >>> t = np.arange(5)
    >>> a = [1+0j, 0.5+0j]
    >>> omegas = [-0.1+0.5j, -0.2-0.3j]
    >>> y = generate_clean_data(t, a, omegas, dt=1.0)
    >>> y.shape
    (5,)
    """
    a = np.asarray(a, dtype=complex)
    omegas = np.asarray(omegas, dtype=complex)
    t = np.asarray(t)

    if a.ndim != 1 or omegas.ndim != 1:
        raise ValueError("a and omegas must be 1-D arrays.")
    if len(a) != len(omegas):
        raise ValueError(
            f"a and omegas must have the same length, "
            f"got len(a)={len(a)} and len(omegas)={len(omegas)}."
        )
    if t.size == 0:
        raise ValueError("t must not be empty.")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got dt={dt}.")

    y_clean = np.sum(a[:, None] * np.exp(omegas[:, None] * (t * dt)), axis=0)
    return y_clean


def generate_noisy_data(
    y_clean: np.ndarray,
    noise_level: float,
    rng: np.random.Generator | None = None,
    complex_noise: bool = False
) -> np.ndarray:
    """
    Add Gaussian noise to a clean signal.

    Parameters
    ----------
    y_clean : np.ndarray
        Clean input signal.
    noise_level : float
        Standard deviation of the additive Gaussian noise (must be >= 0).
    rng : numpy.random.Generator, optional
        Random number generator. If None, a new default generator is created.
        Pass an explicit generator for reproducibility.
    complex_noise : bool, optional
        If True and y_clean is complex, add independent Gaussian noise to
        both real and imaginary parts. If False (default), noise is added
        to the real part only.

    Returns
    -------
    y_noisy : np.ndarray
        Noisy signal, same shape as y_clean.

    Raises
    ------
    ValueError
        If noise_level is negative.

    Example
    -------
    >>> y_clean = np.array([1.0, 2.0, 3.0])
    >>> rng = np.random.default_rng(0)
    >>> y_noisy = generate_noisy_data(y_clean, noise_level=0.1, rng=rng)
    >>> y_noisy.shape
    (3,)
    """
    if rng is None:
        rng = np.random.default_rng()
    if noise_level < 0:
        raise ValueError(
            f"noise_level must be non-negative, got noise_level={noise_level}."
        )

    if complex_noise and np.iscomplexobj(y_clean):
        noise_real = rng.normal(0.0, noise_level, size=y_clean.shape)
        noise_imag = rng.normal(0.0, noise_level, size=y_clean.shape)
        gaussian_noise = noise_real + 1j * noise_imag
    else:
        gaussian_noise = rng.normal(0.0, noise_level, size=y_clean.shape)

    return y_clean + gaussian_noise
