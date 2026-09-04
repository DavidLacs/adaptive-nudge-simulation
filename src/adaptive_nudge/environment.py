"""Simulation environment for synthetic session generation and outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import (
    ACTION_SPACES,
    GROUND_TRUTH_COEFFICIENTS,
    SimulationConfig,
    TIME_OF_DAY_PROBABILITIES,
    TIME_OF_DAY_STATES,
)


TimeOfDay = Literal[
    "morning",
    "afternoon",
    "evening",
    "late_night",
]


@dataclass(frozen=True)
class SessionState:
    """State generated before the timing decision."""

    duration_minutes: float
    time_of_day: TimeOfDay
    recent_responsiveness: float
    context: np.ndarray


class Environment:
    """Generate session conditions and evaluate selected timing actions."""

    def __init__(
        self,
        config: SimulationConfig,
        duration_rng: np.random.Generator,
        context_rng: np.random.Generator,
    ) -> None:
        self.config = config
        self._duration_rng = duration_rng
        self._context_rng = context_rng
        self._delivered_outcomes: list[int] = []

        self._post_change_coefficients = (
            self._build_post_change_coefficients()
        )

    def reset(self) -> None:
        """Reset the history of previously delivered intervention outcomes."""
        self._delivered_outcomes.clear()

    def generate_session(self) -> SessionState:
        """Generate session duration and pre-decision context."""
        duration_minutes = float(
            self._duration_rng.lognormal(
                mean=np.log(
                    self.config.session_duration_median_minutes
                ),
                sigma=self.config.session_duration_sigma,
            )
        )

        time_of_day = self._context_rng.choice(
            TIME_OF_DAY_STATES,
            p=TIME_OF_DAY_PROBABILITIES,
        )

        recent_responsiveness = self._recent_responsiveness()

        context = self._build_context(
            time_of_day=time_of_day,
            recent_responsiveness=recent_responsiveness,
        )

        return SessionState(
            duration_minutes=duration_minutes,
            time_of_day=time_of_day,
            recent_responsiveness=recent_responsiveness,
            context=context,
        )

    def evaluate_action(
        self,
        session: SessionState,
        timing_minutes: int,
        response_rng: np.random.Generator,
        decision_index: int,
    ) -> tuple[int, int, float]:
        """Evaluate one selected timing and update response history.

        Returns:
            opportunity: 1 if the selected timing is reached, otherwise 0.
            response: 1 if a delivered intervention receives a response,
                otherwise 0.
            reward: session-level reward.
        """
        valid_timings = {
            timing
            for action_space in ACTION_SPACES.values()
            for timing in action_space
        }

        if timing_minutes not in valid_timings:
            raise ValueError(
                f"Unsupported timing: {timing_minutes}"
            )

        opportunity = int(
            session.duration_minutes >= timing_minutes
        )

        if opportunity == 0:
            return 0, 0, 0.0

        response_probability = self.response_probability(
            timing_minutes=timing_minutes,
            context=session.context,
            decision_index=decision_index,
        )

        response = int(
            response_rng.random() < response_probability
        )

        self._delivered_outcomes.append(response)

        reward = float(response)

        return opportunity, response, reward

    def response_probability(
        self,
        timing_minutes: int,
        context: np.ndarray,
        decision_index: int,
    ) -> float:
        """Compute the ground-truth response probability."""
        coefficients = self._coefficients_for_decision(
            decision_index
        )

        probability = float(
            np.dot(coefficients[timing_minutes], context)
        )

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Response probability must be in [0, 1], "
                f"got {probability}"
            )

        return probability

    def _recent_responsiveness(self) -> float:
        """Compute recent responsiveness from delivered outcomes only."""
        recent = self._delivered_outcomes[
            -self.config.responsiveness_window:
        ]

        if not recent:
            return 0.0

        return float(np.mean(recent))

    @staticmethod
    def _build_context(
        time_of_day: str,
        recent_responsiveness: float,
    ) -> np.ndarray:
        """Construct the five-dimensional context vector."""
        time_index = TIME_OF_DAY_STATES.index(time_of_day)

        temporal_context = np.zeros(
            len(TIME_OF_DAY_STATES),
            dtype=float,
        )
        temporal_context[time_index] = 1.0

        return np.concatenate(
            (
                temporal_context,
                np.array(
                    [recent_responsiveness],
                    dtype=float,
                ),
            )
        )

    def _coefficients_for_decision(
        self,
        decision_index: int,
    ) -> dict[int, tuple[float, float, float, float, float]]:
        """Return the coefficient regime active at the decision index."""
        if decision_index < self.config.regime_change_point:
            return GROUND_TRUTH_COEFFICIENTS

        return self._post_change_coefficients

    @staticmethod
    def _build_post_change_coefficients() -> dict[
        int, tuple[float, float, float, float, float]
    ]:
        """Reverse temporal coefficients while preserving responsiveness."""
        return {
            timing: (
                coefficients[3],
                coefficients[2],
                coefficients[1],
                coefficients[0],
                coefficients[4],
            )
            for timing, coefficients in GROUND_TRUTH_COEFFICIENTS.items()
        }