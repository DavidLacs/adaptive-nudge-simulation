"""Ground-truth evaluation primitives for the adaptive nudge simulation."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from adaptive_nudge.config import SimulationConfig
from adaptive_nudge.environment import Environment


class GroundTruthEvaluator:
    """Evaluate expected rewards using the environment's ground truth."""

    def __init__(
        self,
        config: SimulationConfig,
        non_stationary: bool = False,
    ) -> None:
        self.config = config

        # The evaluator uses the Environment only for its deterministic
        # ground-truth coefficient regime. These RNGs are never used.
        self._environment = Environment(
            config=config,
            duration_rng=np.random.default_rng(0),
            context_rng=np.random.default_rng(1),
            non_stationary=non_stationary,
        )

        # P(D >= a) depends only on the session-duration distribution
        # and timing a. Cache it once for all timings used by the
        # configured action spaces.
        timing_values = sorted(
            {
                timing
                for action_space in (
                    # Importing ACTION_SPACES here would duplicate the
                    # configuration source, so use the supported timing
                    # range defined by the environment indirectly below.
                    ()
                )
                for timing in action_space
            }
        )

        # The experiment action spaces are fixed by the central
        # configuration. Import them locally to keep the evaluator's
        # existing top-level dependency surface unchanged.
        from adaptive_nudge.config import ACTION_SPACES

        timing_values = sorted(
            {
                timing
                for action_space in ACTION_SPACES.values()
                for timing in action_space
            }
        )

        self._opportunity_probabilities = {
            timing: self._compute_probability_of_reaching_timing(
                timing
            )
            for timing in timing_values
        }

    def _compute_probability_of_reaching_timing(
        self,
        timing_minutes: int,
    ) -> float:
        """Compute P(D >= a) under the configured lognormal model."""
        if timing_minutes <= 0:
            raise ValueError(
                "timing_minutes must be positive."
            )

        mu = self.config.session_duration_mu
        sigma = self.config.session_duration_sigma

        if sigma <= 0:
            raise ValueError(
                "session_duration_sigma must be positive."
            )

        z = (
            math.log(timing_minutes) - mu
        ) / (sigma * math.sqrt(2.0))

        return 0.5 * math.erfc(z)

    def probability_of_reaching_timing(
        self,
        timing_minutes: int,
    ) -> float:
        """Return P(D >= a) under the configured lognormal model."""
        if timing_minutes in self._opportunity_probabilities:
            return self._opportunity_probabilities[timing_minutes]

        probability = self._compute_probability_of_reaching_timing(
            timing_minutes
        )

        self._opportunity_probabilities[timing_minutes] = probability

        return probability

    def response_probability(
        self,
        timing_minutes: int,
        context: np.ndarray,
        decision_index: int,
    ) -> float:
        """Return the ground-truth conditional response probability."""
        coefficients = (
            self._environment.coefficients_for_decision(
                decision_index
            )
        )

        if timing_minutes not in coefficients:
            raise ValueError(
                f"Unsupported timing: {timing_minutes}"
            )

        context_array = np.asarray(
            context,
            dtype=float,
        )

        if context_array.shape != (
            self.config.context_dimension,
        ):
            raise ValueError(
                "context must have shape "
                f"({self.config.context_dimension},)."
            )

        probability = float(
            np.dot(
                coefficients[timing_minutes],
                context_array,
            )
        )

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "Response probability must be in [0, 1], "
                f"got {probability}"
            )

        return probability

    def expected_reward(
        self,
        timing_minutes: int,
        context: np.ndarray,
        decision_index: int,
    ) -> float:
        """Return the ground-truth expected session-level reward."""
        opportunity_probability = (
            self.probability_of_reaching_timing(
                timing_minutes
            )
        )

        conditional_response_probability = (
            self.response_probability(
                timing_minutes,
                context,
                decision_index,
            )
        )

        return (
            opportunity_probability
            * conditional_response_probability
        )

    def expected_rewards(
        self,
        context: np.ndarray,
        decision_index: int,
        actions: Sequence[int],
    ) -> tuple[tuple[int, ...], np.ndarray]:
        """Return available actions and their expected rewards."""
        available_actions = tuple(sorted(set(actions)))

        if not available_actions:
            raise ValueError(
                "actions must contain at least one timing."
            )

        rewards = np.array(
            [
                self.expected_reward(
                    timing_minutes=timing,
                    context=context,
                    decision_index=decision_index,
                )
                for timing in available_actions
            ],
            dtype=float,
        )

        return available_actions, rewards

    def optimal_actions(
        self,
        context: np.ndarray,
        decision_index: int,
        actions: Sequence[int],
    ) -> tuple[int, ...]:
        """Return all actions attaining maximum expected reward."""
        available_actions, rewards = self.expected_rewards(
            context=context,
            decision_index=decision_index,
            actions=actions,
        )

        maximum_reward = float(
            np.max(rewards)
        )

        return tuple(
            timing
            for timing, reward in zip(
                available_actions,
                rewards,
            )
            if math.isclose(
                reward,
                maximum_reward,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )

    def pseudo_regret(
        self,
        selected_action: int,
        context: np.ndarray,
        decision_index: int,
        actions: Sequence[int],
    ) -> float:
        """Return one-decision ground-truth pseudo-regret."""
        available_actions, rewards = self.expected_rewards(
            context=context,
            decision_index=decision_index,
            actions=actions,
        )

        if selected_action not in available_actions:
            raise ValueError(
                "selected_action must belong to actions."
            )

        selected_index = available_actions.index(
            selected_action
        )

        selected_reward = float(
            rewards[selected_index]
        )

        optimal_reward = float(
            np.max(rewards)
        )

        return max(
            0.0,
            optimal_reward - selected_reward,
        )

    def is_optimal_action(
        self,
        selected_action: int,
        context: np.ndarray,
        decision_index: int,
        actions: Sequence[int],
    ) -> bool:
        """Return whether the selected action is optimal."""
        return selected_action in self.optimal_actions(
            context=context,
            decision_index=decision_index,
            actions=actions,
        )


def optimal_action_indicators(
    records: Sequence,
    actions: Sequence[int],
    evaluator: GroundTruthEvaluator,
) -> np.ndarray:
    """Return 1/0 indicators for optimal-arm selection."""
    indicators = np.empty(
        len(records),
        dtype=np.int8,
    )

    for index, record in enumerate(records):
        indicators[index] = int(
            evaluator.is_optimal_action(
                selected_action=record.timing_minutes,
                context=record.context,
                decision_index=record.decision_index,
                actions=actions,
            )
        )

    return indicators


def rolling_optimal_selection_rate(
    indicators: np.ndarray,
    window: int,
) -> np.ndarray:
    """Return the rolling optimal-arm selection rate."""
    if window <= 0:
        raise ValueError(
            "window must be positive."
        )

    if indicators.ndim != 1:
        raise ValueError(
            "indicators must be one-dimensional."
        )

    if len(indicators) < window:
        return np.array([], dtype=float)

    cumulative = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(
                indicators,
                dtype=np.int64,
            ),
        )
    )

    window_counts = (
        cumulative[window:]
        - cumulative[:-window]
    )

    return window_counts / window


def convergence_index(
    indicators: np.ndarray,
    window: int,
    threshold: float,
) -> int | None:
    """Return the first decision index meeting convergence criteria."""
    if window <= 0:
        raise ValueError(
            "window must be positive."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be in [0, 1]."
        )

    if indicators.ndim != 1:
        raise ValueError(
            "indicators must be one-dimensional."
        )

    rolling_rates = rolling_optimal_selection_rate(
        indicators=indicators,
        window=window,
    )

    qualifying = np.flatnonzero(
        rolling_rates >= threshold
    )

    if len(qualifying) == 0:
        return None

    return int(
        qualifying[0] + window - 1
    )


def recovery_index(
    indicators: np.ndarray,
    regime_change_point: int,
    window: int,
    threshold: float,
) -> int | None:
    """Return the first qualifying post-change recovery index."""
    if regime_change_point < 0:
        raise ValueError(
            "regime_change_point must be non-negative."
        )

    if window <= 0:
        raise ValueError(
            "window must be positive."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be in [0, 1]."
        )

    if indicators.ndim != 1:
        raise ValueError(
            "indicators must be one-dimensional."
        )

    rolling_rates = rolling_optimal_selection_rate(
        indicators=indicators,
        window=window,
    )

    first_valid_index = (
        regime_change_point + window
    )

    qualifying_indices = np.flatnonzero(
        rolling_rates >= threshold
    )

    for rolling_index in qualifying_indices:
        decision_index = (
            int(rolling_index) + window - 1
        )

        if decision_index >= first_valid_index:
            return decision_index

    return None


def recovery_delay(
    recovery_index_value: int | None,
    regime_change_point: int,
) -> int | None:
    """Return recovery delay measured from the regime change."""
    if recovery_index_value is None:
        return None

    if recovery_index_value < regime_change_point:
        raise ValueError(
            "recovery_index must not precede "
            "the regime change."
        )

    return (
        recovery_index_value
        - regime_change_point
    )