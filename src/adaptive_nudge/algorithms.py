"""Learning algorithms and fixed baseline for adaptive nudge timing."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseAlgorithm(ABC):
    """Common interface for all timing-selection strategies."""

    def __init__(self, actions: tuple[int, ...]) -> None:
        if not actions:
            raise ValueError("At least one action is required.")

        if tuple(actions) != tuple(sorted(actions)):
            raise ValueError("Actions must be in ascending order.")

        if len(set(actions)) != len(actions):
            raise ValueError("Actions must be unique.")

        self.actions = tuple(actions)

    @abstractmethod
    def select_action(self, context: np.ndarray) -> int:
        """Select one timing action using the current information state."""

    @abstractmethod
    def update(
        self,
        action: int,
        context: np.ndarray,
        reward: float,
    ) -> None:
        """Update the algorithm using the observed session-level reward."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the algorithm to its initial information state."""


class LinUCB(BaseAlgorithm):
    """Disjoint LinUCB for contextual timing selection."""

    def __init__(
        self,
        actions: tuple[int, ...],
        context_dimension: int,
        alpha: float,
    ) -> None:
        super().__init__(actions)

        if context_dimension <= 0:
            raise ValueError("context_dimension must be positive.")

        if alpha < 0.0:
            raise ValueError("alpha must be non-negative.")

        self.context_dimension = context_dimension
        self.alpha = float(alpha)

        self.reset()

    def reset(self) -> None:
        """Initialize Aa = I and ba = 0 independently for each arm."""
        identity = np.eye(self.context_dimension)

        self.A = {
            action: identity.copy()
            for action in self.actions
        }

        self.b = {
            action: np.zeros(self.context_dimension)
            for action in self.actions
        }

    def _validate_context(self, context: np.ndarray) -> np.ndarray:
        """Validate and return the context as a one-dimensional array."""
        context = np.asarray(context, dtype=float)

        if context.shape != (self.context_dimension,):
            raise ValueError(
                "Context must have shape "
                f"({self.context_dimension},), got {context.shape}."
            )

        return context

    def _validate_action(self, action: int) -> None:
        """Ensure the action belongs to this algorithm's action space."""
        if action not in self.actions:
            raise ValueError(f"Unsupported action: {action}")

    def select_action(self, context: np.ndarray) -> int:
        """Select the action with the highest LinUCB score."""
        context = self._validate_context(context)

        scores = []

        for action in self.actions:
            A = self.A[action]
            b = self.b[action]

            theta_hat = np.linalg.solve(A, b)
            A_inv_x = np.linalg.solve(A, context)

            estimated_reward = float(context @ theta_hat)
            uncertainty = float(
                np.sqrt(max(0.0, context @ A_inv_x))
            )

            score = (
                estimated_reward
                + self.alpha * uncertainty
            )

            scores.append(score)

        # np.argmax returns the first maximum, implementing the
        # predefined ascending-order tie-breaking rule.
        return self.actions[int(np.argmax(scores))]

    def update(
        self,
        action: int,
        context: np.ndarray,
        reward: float,
    ) -> None:
        """Update only the selected arm using xt and Rt."""
        self._validate_action(action)
        context = self._validate_context(context)

        reward = float(reward)

        if reward not in (0.0, 1.0):
            raise ValueError(
                f"Reward must be 0.0 or 1.0, got {reward}."
            )

        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context


class UCB1(BaseAlgorithm):
    """Non-contextual UCB1 timing-selection baseline."""

    def __init__(self, actions: tuple[int, ...]) -> None:
        super().__init__(actions)
        self.reset()

    def reset(self) -> None:
        """Reset arm counts and cumulative rewards."""
        self.counts = {
            action: 0
            for action in self.actions
        }

        self.reward_sums = {
            action: 0.0
            for action in self.actions
        }

        self._initialization_index = 0
        self._decision_count = 0

    def _validate_action(self, action: int) -> None:
        """Ensure the action belongs to this algorithm's action space."""
        if action not in self.actions:
            raise ValueError(f"Unsupported action: {action}")

    def select_action(self, context: np.ndarray) -> int:
        """Select an action using UCB1.

        Context is accepted to maintain the common algorithm
        interface, but UCB1 does not use it.
        """
        del context

        # Each available arm is selected once during initialization.
        if self._initialization_index < len(self.actions):
            action = self.actions[self._initialization_index]
            self._initialization_index += 1
            return action

        t = self._decision_count + 1

        scores = []

        for action in self.actions:
            count = self.counts[action]
            reward_sum = self.reward_sums[action]

            mean_reward = reward_sum / count

            exploration = np.sqrt(
                (2.0 * np.log(t)) / count
            )

            scores.append(mean_reward + exploration)

        # First maximum implements ascending-order tie-breaking.
        return self.actions[int(np.argmax(scores))]

    def update(
        self,
        action: int,
        context: np.ndarray,
        reward: float,
    ) -> None:
        """Update only the selected arm using the observed reward."""
        del context

        self._validate_action(action)

        reward = float(reward)

        if reward not in (0.0, 1.0):
            raise ValueError(
                f"Reward must be 0.0 or 1.0, got {reward}."
            )

        self.counts[action] += 1
        self.reward_sums[action] += reward
        self._decision_count += 1


class Static10MinuteBaseline(BaseAlgorithm):
    """Fixed non-adaptive baseline using a 10-minute timing."""

    def __init__(self, actions: tuple[int, ...]) -> None:
        super().__init__(actions)

        if 10 not in self.actions:
            raise ValueError(
                "The 10-minute timing must be available "
                "for the static baseline."
            )

    def select_action(self, context: np.ndarray) -> int:
        """Always select the prespecified 10-minute timing."""
        del context
        return 10

    def update(
        self,
        action: int,
        context: np.ndarray,
        reward: float,
    ) -> None:
        """Do nothing because the static baseline does not learn."""
        del action, context, reward

    def reset(self) -> None:
        """The static baseline has no learned state to reset."""