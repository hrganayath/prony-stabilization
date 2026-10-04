import numpy as np


# ------------------------------------------------------------
# Main paired oversampling experiment
# ------------------------------------------------------------

MODEL_ORDER = 4

RHO_VALUES = np.arange(1, 11)

NOISE_LEVELS = np.linspace(0.0, 0.5, 10)

N_TRIALS = 1000

MAX_RHO = int(RHO_VALUES.max())

MAX_SAMPLES = MODEL_ORDER * (MAX_RHO + 1) + 1


# ------------------------------------------------------------
# Random-number protocol
# ------------------------------------------------------------

# Trial i uses seed i.
# Within a trial, the same length-MAX_SAMPLES noise realization
# is reused across all rho values by taking progressively longer
# prefixes.
TRIAL_SEEDS = np.arange(N_TRIALS, dtype=int)


# ------------------------------------------------------------
# Common scoring window
# ------------------------------------------------------------

SCORING_LENGTH = 2 * MODEL_ORDER + 1


def sample_count(rho: int) -> int:
    """Number of samples used for a given oversampling factor."""
    return MODEL_ORDER * (rho + 1) + 1


def hankel_shape(rho: int) -> tuple[int, int]:
    """Shape of the Hankel matrix used in the main experiment."""
    rows = rho * MODEL_ORDER + 1
    cols = MODEL_ORDER + 1
    return rows, cols
