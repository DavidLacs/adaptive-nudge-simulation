"""Experiment-condition definitions for the adaptive nudge simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from adaptive_nudge.config import (
    ACTION_SPACES,
    SimulationConfig,
)


AlgorithmName = Literal[
    "LinUCB",
    "UCB1",
    "Static10MinuteBaseline",
]

ExperimentType = Literal[
    "primary",
    "action_space",
    "non_stationary",
    "sensitivity",
]

SensitivityParameter = Literal[
    "session_duration_sigma",
    "responsiveness_window",
    "linucb_alpha",
    "temporal_coefficient_scaling",
]


@dataclass(frozen=True)
class ExperimentCondition:
    """Configuration describing one experimental condition."""

    name: str
    experiment_type: ExperimentType
    algorithm: AlgorithmName
    action_space_size: int
    sensitivity_parameter: SensitivityParameter | None = None
    sensitivity_value: float | int | None = None

    def __post_init__(self) -> None:
        """Validate the condition definition."""

        if self.action_space_size not in ACTION_SPACES:
            raise ValueError(
                f"Unsupported action-space size: "
                f"{self.action_space_size}"
            )

        if self.algorithm == "Static10MinuteBaseline":
            if 10 not in ACTION_SPACES[self.action_space_size]:
                raise ValueError(
                    "The static 10-minute baseline requires an "
                    "action space containing 10 minutes."
                )

        if self.experiment_type == "sensitivity":
            if self.sensitivity_parameter is None:
                raise ValueError(
                    "Sensitivity conditions require a "
                    "sensitivity parameter."
                )

            if self.sensitivity_value is None:
                raise ValueError(
                    "Sensitivity conditions require a "
                    "sensitivity value."
                )

        elif (
            self.sensitivity_parameter is not None
            or self.sensitivity_value is not None
        ):
            raise ValueError(
                "Sensitivity parameter and value are only valid "
                "for sensitivity conditions."
            )

    @property
    def actions(self) -> tuple[int, ...]:
        """Return the timing actions for this condition."""

        return tuple(
            ACTION_SPACES[self.action_space_size]
        )


def primary_conditions() -> tuple[ExperimentCondition, ...]:
    """Return the primary three-strategy comparison conditions."""

    return (
        ExperimentCondition(
            name="LinUCB-A5",
            experiment_type="primary",
            algorithm="LinUCB",
            action_space_size=5,
        ),
        ExperimentCondition(
            name="UCB1-A5",
            experiment_type="primary",
            algorithm="UCB1",
            action_space_size=5,
        ),
        ExperimentCondition(
            name="Static10-A5",
            experiment_type="primary",
            algorithm="Static10MinuteBaseline",
            action_space_size=5,
        ),
    )


def action_space_conditions() -> tuple[ExperimentCondition, ...]:
    """Return adaptive-algorithm action-space conditions."""

    conditions: list[ExperimentCondition] = []

    for action_space_size in (3, 5, 8):
        for algorithm in ("LinUCB", "UCB1"):
            conditions.append(
                ExperimentCondition(
                    name=(
                        f"{algorithm}-A{action_space_size}"
                    ),
                    experiment_type="action_space",
                    algorithm=algorithm,
                    action_space_size=action_space_size,
                )
            )

    return tuple(conditions)


def non_stationary_conditions() -> tuple[ExperimentCondition, ...]:
    """Return adaptive-algorithm non-stationary conditions."""

    return (
        ExperimentCondition(
            name="LinUCB-NonStationary-A5",
            experiment_type="non_stationary",
            algorithm="LinUCB",
            action_space_size=5,
        ),
        ExperimentCondition(
            name="UCB1-NonStationary-A5",
            experiment_type="non_stationary",
            algorithm="UCB1",
            action_space_size=5,
        ),
    )


def sensitivity_conditions() -> tuple[ExperimentCondition, ...]:
    """Return one-factor-at-a-time sensitivity conditions at A5.

    Baseline values are excluded because they are already represented
    by the primary A5 conditions. Each sensitivity parameter is varied
    independently while the remaining simulation settings retain their
    baseline values.
    """

    sensitivity_design = (
        (
            "session_duration_sigma",
            (0.75, 1.05),
            "Sigma",
            (
                "LinUCB",
                "UCB1",
                "Static10MinuteBaseline",
            ),
        ),
        (
            "responsiveness_window",
            (3, 10),
            "Window",
            (
                "LinUCB",
                "UCB1",
                "Static10MinuteBaseline",
            ),
        ),
        (
            "linucb_alpha",
            (0.5, 2.0),
            "Alpha",
            ("LinUCB",),
        ),
        (
            "temporal_coefficient_scaling",
            (0.8, 1.2),
            "TemporalScale",
            (
                "LinUCB",
                "UCB1",
                "Static10MinuteBaseline",
            ),
        ),
    )

    conditions: list[ExperimentCondition] = []

    for (
        parameter,
        values,
        parameter_label,
        algorithms,
    ) in sensitivity_design:
        for value in values:
            value_label = str(value).replace(
                ".",
                "p",
            )

            for algorithm in algorithms:
                conditions.append(
                    ExperimentCondition(
                        name=(
                            f"{parameter_label}-{value_label}-"
                            f"{algorithm}-A5"
                        ),
                        experiment_type="sensitivity",
                        algorithm=algorithm,
                        action_space_size=5,
                        sensitivity_parameter=parameter,
                        sensitivity_value=value,
                    )
                )

    return tuple(conditions)


def baseline_config() -> SimulationConfig:
    """Return the default simulation configuration."""

    return SimulationConfig()


def all_experiment_conditions() -> tuple[ExperimentCondition, ...]:
    """Return all non-sensitivity experimental conditions."""

    return (
        primary_conditions()
        + action_space_conditions()
        + non_stationary_conditions()
    )


def all_conditions() -> tuple[ExperimentCondition, ...]:
    """Return all defined experimental conditions, including sensitivity."""

    return (
        all_experiment_conditions()
        + sensitivity_conditions()
    )