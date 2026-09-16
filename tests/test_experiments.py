"""Tests for experiment-condition definitions."""

from __future__ import annotations

import pytest

from adaptive_nudge.config import ACTION_SPACES
from adaptive_nudge.experiments import (
    ExperimentCondition,
    action_space_conditions,
    all_conditions,
    all_experiment_conditions,
    non_stationary_conditions,
    primary_conditions,
    sensitivity_conditions,
)


def test_experiment_condition_exposes_configured_actions() -> None:
    """Verify an experiment condition resolves its action space."""

    condition = ExperimentCondition(
        name="test",
        experiment_type="primary",
        algorithm="LinUCB",
        action_space_size=5,
    )

    assert condition.actions == ACTION_SPACES[5]


def test_primary_conditions_match_thesis_design() -> None:
    """Verify the primary comparison contains the three specified strategies."""

    conditions = primary_conditions()

    assert len(conditions) == 3

    assert {
        condition.algorithm
        for condition in conditions
    } == {
        "LinUCB",
        "UCB1",
        "Static10MinuteBaseline",
    }

    assert all(
        condition.action_space_size == 5
        for condition in conditions
    )

    assert all(
        condition.experiment_type == "primary"
        for condition in conditions
    )


def test_action_space_conditions_match_thesis_design() -> None:
    """Verify three-, five-, and eight-arm adaptive conditions."""

    conditions = action_space_conditions()

    assert len(conditions) == 6

    assert {
        condition.action_space_size
        for condition in conditions
    } == {3, 5, 8}

    assert {
        condition.algorithm
        for condition in conditions
    } == {"LinUCB", "UCB1"}

    assert all(
        condition.experiment_type == "action_space"
        for condition in conditions
    )


def test_non_stationary_conditions_use_adaptive_algorithms() -> None:
    """Verify non-stationary conditions contain the learning algorithms."""

    conditions = non_stationary_conditions()

    assert len(conditions) == 2

    assert {
        condition.algorithm
        for condition in conditions
    } == {"LinUCB", "UCB1"}

    assert all(
        condition.action_space_size == 5
        for condition in conditions
    )

    assert all(
        condition.experiment_type == "non_stationary"
        for condition in conditions
    )


def test_sensitivity_conditions_match_thesis_design() -> None:
    """Verify the OFAT sensitivity analysis uses non-baseline A5 conditions."""

    conditions = sensitivity_conditions()

    assert len(conditions) == 20

    assert {
        condition.sensitivity_parameter
        for condition in conditions
    } == {
        "session_duration_sigma",
        "responsiveness_window",
        "linucb_alpha",
        "temporal_coefficient_scaling",
    }

    assert all(
        condition.experiment_type == "sensitivity"
        for condition in conditions
    )

    assert all(
        condition.sensitivity_parameter is not None
        and condition.sensitivity_value is not None
        for condition in conditions
    )

    assert {
        condition.action_space_size
        for condition in conditions
    } == {5}

    assert {
        condition.algorithm
        for condition in conditions
    } == {
        "LinUCB",
        "UCB1",
        "Static10MinuteBaseline",
    }


def test_sensitivity_values_match_thesis_design() -> None:
    """Verify only non-baseline sensitivity values are represented."""

    conditions = sensitivity_conditions()

    values_by_parameter = {
        parameter: {
            condition.sensitivity_value
            for condition in conditions
            if condition.sensitivity_parameter == parameter
        }
        for parameter in (
            "session_duration_sigma",
            "responsiveness_window",
            "linucb_alpha",
            "temporal_coefficient_scaling",
        )
    }

    assert values_by_parameter == {
        "session_duration_sigma": {0.75, 1.05},
        "responsiveness_window": {3, 10},
        "linucb_alpha": {0.5, 2.0},
        "temporal_coefficient_scaling": {0.8, 1.2},
    }


def test_sensitivity_algorithm_counts_match_design() -> None:
    """Verify each sensitivity parameter uses only applicable algorithms."""

    conditions = sensitivity_conditions()

    for parameter in (
        "session_duration_sigma",
        "responsiveness_window",
        "temporal_coefficient_scaling",
    ):
        parameter_conditions = [
            condition
            for condition in conditions
            if condition.sensitivity_parameter == parameter
        ]

        assert len(parameter_conditions) == 6

        assert {
            condition.algorithm
            for condition in parameter_conditions
        } == {
            "LinUCB",
            "UCB1",
            "Static10MinuteBaseline",
        }

    alpha_conditions = [
        condition
        for condition in conditions
        if condition.sensitivity_parameter == "linucb_alpha"
    ]

    assert len(alpha_conditions) == 2

    assert all(
        condition.algorithm == "LinUCB"
        for condition in alpha_conditions
    )


def test_sensitivity_conditions_are_a5_only() -> None:
    """Verify sensitivity analysis does not vary action-space size."""

    conditions = sensitivity_conditions()

    assert all(
        condition.action_space_size == 5
        for condition in conditions
    )


def test_sensitivity_baseline_values_are_excluded() -> None:
    """Verify primary baseline values are not duplicated in sensitivity runs."""

    conditions = sensitivity_conditions()

    assert all(
        condition.sensitivity_value != 0.90
        for condition in conditions
        if condition.sensitivity_parameter
        == "session_duration_sigma"
    )

    assert all(
        condition.sensitivity_value != 5
        for condition in conditions
        if condition.sensitivity_parameter
        == "responsiveness_window"
    )

    assert all(
        condition.sensitivity_value != 1.0
        for condition in conditions
        if condition.sensitivity_parameter
        == "linucb_alpha"
    )

    assert all(
        condition.sensitivity_value != 1.0
        for condition in conditions
        if condition.sensitivity_parameter
        == "temporal_coefficient_scaling"
    )


def test_sensitivity_condition_requires_parameter_and_value() -> None:
    """Verify sensitivity conditions cannot omit their varied setting."""

    with pytest.raises(ValueError):
        ExperimentCondition(
            name="invalid-sensitivity",
            experiment_type="sensitivity",
            algorithm="LinUCB",
            action_space_size=5,
        )


def test_non_sensitivity_condition_cannot_define_sensitivity_value() -> None:
    """Verify sensitivity metadata is restricted to sensitivity conditions."""

    with pytest.raises(ValueError):
        ExperimentCondition(
            name="invalid-primary",
            experiment_type="primary",
            algorithm="LinUCB",
            action_space_size=5,
            sensitivity_parameter="linucb_alpha",
            sensitivity_value=0.5,
        )


def test_all_experiment_conditions_exclude_sensitivity_conditions() -> None:
    """Verify the standard experiment suite excludes sensitivity analysis."""

    conditions = all_experiment_conditions()

    assert len(conditions) == 11

    assert all(
        condition.experiment_type != "sensitivity"
        for condition in conditions
    )


def test_all_conditions_include_sensitivity_conditions() -> None:
    """Verify the complete condition registry contains all 31 conditions."""

    conditions = all_conditions()

    assert len(conditions) == 31

    assert sum(
        condition.experiment_type == "primary"
        for condition in conditions
    ) == 3

    assert sum(
        condition.experiment_type == "action_space"
        for condition in conditions
    ) == 6

    assert sum(
        condition.experiment_type == "non_stationary"
        for condition in conditions
    ) == 2

    assert sum(
        condition.experiment_type == "sensitivity"
        for condition in conditions
    ) == 20