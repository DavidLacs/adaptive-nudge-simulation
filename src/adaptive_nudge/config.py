"""Central configuration for the adaptive nudge simulation."""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

BASE_SEED = 20260903

RNG_BIT_GENERATOR = "PCG64"

RNG_STREAMS = (
    "duration",
    "context",
    "linucb_response",
    "ucb1_response",
    "static_response",
)


# ---------------------------------------------------------------------------
# Simulation budget
# ---------------------------------------------------------------------------

DECISIONS_PER_REPLICATION = 10_000
NUM_REPLICATIONS = 100


# ---------------------------------------------------------------------------
# Session duration
# ---------------------------------------------------------------------------

SESSION_DURATION_MEDIAN_MINUTES = 8.5
SESSION_DURATION_SIGMA = 0.90


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

TIME_OF_DAY_STATES = (
    "morning",
    "afternoon",
    "evening",
    "late_night",
)

TIME_OF_DAY_PROBABILITIES = (
    0.25,
    0.25,
    0.25,
    0.25,
)

RESPONSIVENESS_WINDOW = 5


# ---------------------------------------------------------------------------
# Timing action spaces
# ---------------------------------------------------------------------------

ACTION_SPACES = {
    3: (3, 10, 20),
    5: (3, 7, 10, 15, 20),
    8: (3, 5, 7, 10, 13, 15, 18, 20),
}

MIN_TIMING_MINUTES = 3
MAX_TIMING_MINUTES = 20
STATIC_TIMING_MINUTES = 10


# ---------------------------------------------------------------------------
# Ground-truth response coefficients
#
# Columns:
#   morning, afternoon, evening, late_night, recent_responsiveness
# ---------------------------------------------------------------------------

GROUND_TRUTH_COEFFICIENTS = {
    3: (0.45, 0.30, 0.15, 0.08, 0.03),
    5: (0.38, 0.40, 0.20, 0.10, 0.03),
    7: (0.30, 0.50, 0.25, 0.12, 0.03),
    10: (0.20, 0.35, 0.50, 0.20, 0.03),
    13: (0.15, 0.35, 0.48, 0.35, 0.03),
    15: (0.10, 0.40, 0.45, 0.45, 0.03),
    18: (0.08, 0.30, 0.40, 0.60, 0.03),
    20: (0.08, 0.25, 0.35, 0.72, 0.03),
}


# ---------------------------------------------------------------------------
# LinUCB
# ---------------------------------------------------------------------------

LINUCB_ALPHA = 1.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

CONVERGENCE_WINDOW = 1_000
CONVERGENCE_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Non-stationarity
# ---------------------------------------------------------------------------

REGIME_CHANGE_FRACTION = 0.5


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

DURATION_SIGMA_SENSITIVITY = (0.75, 0.90, 1.05)

RESPONSIVENESS_WINDOW_SENSITIVITY = (3, 5, 10)

LINUCB_ALPHA_SENSITIVITY = (0.5, 1.0, 2.0)

TEMPORAL_COEFFICIENT_SCALING_SENSITIVITY = (
    0.8,
    1.0,
    1.2,
)

RECENT_RESPONSIVENESS_COEFFICIENT = 0.03


@dataclass(frozen=True)
class SimulationConfig:
    """Resolved configuration for one simulation condition."""

    decisions_per_replication: int = (
        DECISIONS_PER_REPLICATION
    )
    num_replications: int = NUM_REPLICATIONS

    session_duration_median_minutes: float = (
        SESSION_DURATION_MEDIAN_MINUTES
    )
    session_duration_sigma: float = (
        SESSION_DURATION_SIGMA
    )

    responsiveness_window: int = (
        RESPONSIVENESS_WINDOW
    )

    linucb_alpha: float = LINUCB_ALPHA

    temporal_coefficient_scaling: float = 1.0

    convergence_window: int = (
        CONVERGENCE_WINDOW
    )
    convergence_threshold: float = (
        CONVERGENCE_THRESHOLD
    )

    regime_change_fraction: float = (
        REGIME_CHANGE_FRACTION
    )

    base_seed: int = BASE_SEED

    @property
    def context_dimension(self) -> int:
        """Number of components in the context vector."""
        return len(TIME_OF_DAY_STATES) + 1

    @property
    def regime_change_point(self) -> int:
        """Decision index at which the non-stationary regime changes."""
        return int(
            self.decisions_per_replication
            * self.regime_change_fraction
        )

    @property
    def session_duration_mu(self) -> float:
        """Lognormal location parameter derived from the specified median."""
        import math

        return math.log(
            self.session_duration_median_minutes
        )