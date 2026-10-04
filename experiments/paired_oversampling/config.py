import numpy as np


# ============================================================
# Main paired oversampling experiment
# ============================================================

MODEL_ORDER = 4

RHO_VALUES = np.arange(1, 11)

NOISE_LEVELS = np.linspace(0.0, 0.5, 10)

N_TRIALS = 1000


# ============================================================
# Ground-truth n=4 damped exponential model
# ============================================================

AMPLITUDES = np.array(
    [1.0, 1.1, 0.9, 0.7],
    dtype=np.complex128,
)

DAMPING = np.array(
    [-0.10, -0.07, -0.12, -0.08],
    dtype=float,
)

FREQUENCIES = np.array(
    [0.10, -0.22, 0.33, -0.41],
    dtype=float,
)

ANGULAR_FREQUENCIES = 2.0 * np.pi * FREQUENCIES

EXPONENTS = DAMPING + 1j * ANGULAR_FREQUENCIES

POLES = np.exp(EXPONENTS)


# ============================================================
# Maximum signal length required
# ============================================================

MAX_RHO = int(RHO_VALUES.max())

MAX_SAMPLES = MODEL_ORDER * (MAX_RHO + 1) + 1


# ============================================================
# Random-number protocol
# ============================================================

# Trial i uses seed i.
#
# Within one trial, a single noise realization of length
# MAX_SAMPLES is generated. Different rho values use
# progressively longer prefixes of that same realization.
#
# This gives paired comparisons across rho.

TRIAL_SEEDS = np.arange(N_TRIALS, dtype=int)


# ============================================================
# Common reconstruction scoring window
# ============================================================

# All clean/noisy reconstruction comparisons are evaluated
# on the same first 2n+1 samples.

SCORING_LENGTH = 2 * MODEL_ORDER + 1


# ============================================================
# Helper functions
# ============================================================

def sample_count(rho: int) -> int:
    """
    Return the number of samples used for oversampling factor rho.

    N = n(rho + 1) + 1
    """
    return MODEL_ORDER * (rho + 1) + 1


def hankel_shape(rho: int) -> tuple[int, int]:
    """
    Return the Hankel matrix shape for oversampling factor rho.

    H_rho has shape:
        (rho*n + 1) x (n + 1)
    """
    rows = rho * MODEL_ORDER + 1
    cols = MODEL_ORDER + 1

    return rows, cols


def clean_signal(length: int) -> np.ndarray:
    """
    Generate the exact clean exponential signal

        s_k = sum_j A_j exp(lambda_j k)

    for k = 0, ..., length-1.
    """
    k = np.arange(length, dtype=float)

    vandermonde = np.exp(
        np.outer(k, EXPONENTS)
    )

    return vandermonde @ AMPLITUDES
