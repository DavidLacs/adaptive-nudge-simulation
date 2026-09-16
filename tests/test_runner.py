"""Tests for experiment-level simulation orchestration."""

from __future__ import annotations

from dataclasses import replace

from adaptive_nudge.environment import Environment

import pytest

from adaptive_nudge.config import (
    ACTION_SPACES,
    SimulationConfig,
)
from adaptive_nudge.experiments import (
    ExperimentCondition,
    all_experiment_conditions,
)
from adaptive_nudge.runner import (
    ExperimentResult,
    _config_for_condition,
    run_experiment_condition,
    run_experiment_suite,
)


def _small_config() -> SimulationConfig:
    """Return a small deterministic configuration for fast tests."""

    return replace(
        SimulationConfig(),
        decisions_per_replication=20,
        num_replications=3,
        convergence_window=5,
    )


def test_run_experiment_condition_returns_expected_structure() -> None:
    """Verify one experiment condition produces the expected results."""

    config = _small_config()

    condition = ExperimentCondition(
        name="LinUCB-A5",
        experiment_type="primary",
        algorithm="LinUCB",
        action_space_size=5,
    )

    result = run_experiment_condition(
        config=config,
        condition=condition,
    )

    assert isinstance(
        result,
        ExperimentResult,
    )

    assert result.condition == condition

    assert result.condition.algorithm == "LinUCB"

    assert result.condition.action_space_size == 5

    assert result.condition.actions == ACTION_SPACES[5]

    assert len(
        result.replication_metrics
    ) == config.num_replications

    assert {
        metric.replication
        for metric in result.replication_metrics
    } == {0, 1, 2}

    assert all(
        metric.num_decisions
        == config.decisions_per_replication
        for metric in result.replication_metrics
    )

    assert "cumulative_pseudo_regret" in (
        result.aggregated_metrics
    )

    assert "mean_pseudo_regret" in (
        result.aggregated_metrics
    )

    assert "successful_session_exit_rate" in (
        result.aggregated_metrics
    )

    assert "optimal_arm_selection_rate" in (
        result.aggregated_metrics
    )


def test_static_baseline_condition_is_supported() -> None:
    """Verify the fixed 10-minute baseline is a supported condition."""

    config = _small_config()

    condition = ExperimentCondition(
        name="Static10-A5",
        experiment_type="primary",
        algorithm="Static10MinuteBaseline",
        action_space_size=5,
    )

    result = run_experiment_condition(
        config=config,
        condition=condition,
    )

    assert result.condition == condition

    assert result.condition.algorithm == (
        "Static10MinuteBaseline"
    )

    assert result.condition.action_space_size == 5

    assert result.condition.actions == ACTION_SPACES[5]

    assert 10 in result.condition.actions

    assert len(
        result.replication_metrics
    ) == config.num_replications


def test_experiment_condition_is_deterministic() -> None:
    """Verify repeated execution produces identical results."""

    config = _small_config()

    condition = ExperimentCondition(
        name="LinUCB-A5",
        experiment_type="primary",
        algorithm="LinUCB",
        action_space_size=5,
    )

    first = run_experiment_condition(
        config=config,
        condition=condition,
    )

    second = run_experiment_condition(
        config=config,
        condition=condition,
    )

    assert first.condition == second.condition

    assert first.replication_metrics == (
        second.replication_metrics
    )

    assert first.aggregated_metrics == (
        second.aggregated_metrics
    )


def test_paired_algorithms_use_same_replication_schedule() -> None:
    """Verify paired conditions use the same replication structure."""

    config = _small_config()

    linucb_condition = ExperimentCondition(
        name="LinUCB-A5",
        experiment_type="primary",
        algorithm="LinUCB",
        action_space_size=5,
    )

    ucb1_condition = ExperimentCondition(
        name="UCB1-A5",
        experiment_type="primary",
        algorithm="UCB1",
        action_space_size=5,
    )

    linucb = run_experiment_condition(
        config=config,
        condition=linucb_condition,
    )

    ucb1 = run_experiment_condition(
        config=config,
        condition=ucb1_condition,
    )

    assert len(linucb.replication_metrics) == (
        config.num_replications
    )

    assert len(ucb1.replication_metrics) == (
        config.num_replications
    )

    for lin_metric, ucb_metric in zip(
        linucb.replication_metrics,
        ucb1.replication_metrics,
    ):
        assert lin_metric.replication == (
            ucb_metric.replication
        )

        assert lin_metric.num_decisions == (
            ucb_metric.num_decisions
        )


def test_invalid_action_space_is_rejected() -> None:
    """Verify unsupported action-space sizes fail explicitly."""

    with pytest.raises(ValueError):
        ExperimentCondition(
            name="invalid",
            experiment_type="primary",
            algorithm="LinUCB",
            action_space_size=4,
        )


def test_invalid_algorithm_is_rejected() -> None:
    """Verify unsupported algorithm names fail explicitly."""

    config = _small_config()

    class InvalidCondition:
        """Minimal invalid condition for runner validation."""

        algorithm = "UnsupportedAlgorithm"
        actions = ACTION_SPACES[5]

    with pytest.raises(ValueError):
        run_experiment_condition(
            config=config,
            condition=InvalidCondition(),
        )


def test_run_experiment_suite_returns_one_result_per_condition() -> None:
    """Verify the suite runner executes every supplied condition."""

    config = _small_config()

    conditions = (
        ExperimentCondition(
            name="LinUCB-A3",
            experiment_type="action_space",
            algorithm="LinUCB",
            action_space_size=3,
        ),
        ExperimentCondition(
            name="UCB1-A5",
            experiment_type="action_space",
            algorithm="UCB1",
            action_space_size=5,
        ),
    )

    results = run_experiment_suite(
        config=config,
        conditions=conditions,
    )

    assert isinstance(
        results,
        tuple,
    )

    assert len(results) == len(conditions)

    assert tuple(
        result.condition
        for result in results
    ) == conditions

    assert all(
        len(result.replication_metrics)
        == config.num_replications
        for result in results
    )


def test_run_experiment_suite_is_deterministic() -> None:
    """Verify repeated suite execution produces identical results."""

    config = _small_config()

    conditions = (
        ExperimentCondition(
            name="LinUCB-A3",
            experiment_type="action_space",
            algorithm="LinUCB",
            action_space_size=3,
        ),
        ExperimentCondition(
            name="UCB1-A5",
            experiment_type="action_space",
            algorithm="UCB1",
            action_space_size=5,
        ),
    )

    first = run_experiment_suite(
        config=config,
        conditions=conditions,
    )

    second = run_experiment_suite(
        config=config,
        conditions=conditions,
    )

    assert first == second


def test_run_experiment_suite_rejects_empty_conditions() -> None:
    """Verify an empty experiment suite fails explicitly."""

    config = _small_config()

    with pytest.raises(ValueError):
        run_experiment_suite(
            config=config,
            conditions=(),
        )


def test_run_experiment_suite_executes_all_conditions() -> None:
    """Verify the complete non-sensitivity experiment suite executes."""

    config = replace(
        SimulationConfig(),
        decisions_per_replication=5,
        num_replications=2,
        convergence_window=2,
    )

    conditions = all_experiment_conditions()

    assert len(conditions) == 11

    results = run_experiment_suite(
        config=config,
        conditions=conditions,
    )

    assert len(results) == len(conditions)

    assert tuple(
        result.condition.name
        for result in results
    ) == tuple(
        condition.name
        for condition in conditions
    )

    assert all(
        len(result.replication_metrics)
        == config.num_replications
        for result in results
    )

    assert all(
        metric.num_decisions
        == config.decisions_per_replication
        for result in results
        for metric in result.replication_metrics
    )


def test_run_experiment_suite_runs_all_non_sensitivity_conditions() -> None:
    """Verify the complete non-sensitivity experiment suite runs."""

    from adaptive_nudge.experiments import (
        all_experiment_conditions,
    )
    from adaptive_nudge.runner import run_experiment_suite

    config = _small_config()
    conditions = all_experiment_conditions()

    results = run_experiment_suite(
        config=config,
        conditions=conditions,
    )

    assert len(results) == len(conditions)
    assert len(results) == 11

    assert tuple(
        result.condition.name
        for result in results
    ) == tuple(
        condition.name
        for condition in conditions
    )

    assert all(
        len(result.replication_metrics)
        == config.num_replications
        for result in results
    )

    assert all(
        metric.num_decisions
        == config.decisions_per_replication
        for result in results
        for metric in result.replication_metrics
    )


def test_run_experiment_suite_rejects_empty_conditions() -> None:
    """Verify an empty experiment suite is rejected."""

    config = _small_config()

    from adaptive_nudge.runner import run_experiment_suite

    with pytest.raises(ValueError):
        run_experiment_suite(
            config=config,
            conditions=(),
        )

def test_sensitivity_condition_changes_only_selected_config_parameter() -> None:
    """Verify one-factor sensitivity changes only its selected parameter."""

    config = SimulationConfig(
        session_duration_sigma=0.90,
        responsiveness_window=5,
        linucb_alpha=1.0,
        temporal_coefficient_scaling=1.0,
    )

    condition = ExperimentCondition(
        name="Sigma-0p75-LinUCB-A5",
        experiment_type="sensitivity",
        algorithm="LinUCB",
        action_space_size=5,
        sensitivity_parameter="session_duration_sigma",
        sensitivity_value=0.75,
    )

    resolved = _config_for_condition(
        config=config,
        condition=condition,
    )

    assert resolved.session_duration_sigma == 0.75
    assert resolved.responsiveness_window == config.responsiveness_window
    assert resolved.linucb_alpha == config.linucb_alpha
    assert (
        resolved.temporal_coefficient_scaling
        == config.temporal_coefficient_scaling
    )


def test_responsiveness_window_sensitivity_is_applied() -> None:
    """Verify responsiveness-window sensitivity changes the window only."""

    config = SimulationConfig(
        session_duration_sigma=0.90,
        responsiveness_window=5,
        linucb_alpha=1.0,
        temporal_coefficient_scaling=1.0,
    )

    condition = ExperimentCondition(
        name="Window-10-LinUCB-A5",
        experiment_type="sensitivity",
        algorithm="LinUCB",
        action_space_size=5,
        sensitivity_parameter="responsiveness_window",
        sensitivity_value=10,
    )

    resolved = _config_for_condition(
        config=config,
        condition=condition,
    )

    assert resolved.responsiveness_window == 10
    assert resolved.session_duration_sigma == config.session_duration_sigma
    assert resolved.linucb_alpha == config.linucb_alpha
    assert (
        resolved.temporal_coefficient_scaling
        == config.temporal_coefficient_scaling
    )


def test_linucb_alpha_sensitivity_is_applied() -> None:
    """Verify LinUCB alpha sensitivity changes alpha only."""

    config = SimulationConfig(
        session_duration_sigma=0.90,
        responsiveness_window=5,
        linucb_alpha=1.0,
        temporal_coefficient_scaling=1.0,
    )

    condition = ExperimentCondition(
        name="Alpha-2p0-LinUCB-A5",
        experiment_type="sensitivity",
        algorithm="LinUCB",
        action_space_size=5,
        sensitivity_parameter="linucb_alpha",
        sensitivity_value=2.0,
    )

    resolved = _config_for_condition(
        config=config,
        condition=condition,
    )

    assert resolved.linucb_alpha == 2.0
    assert resolved.session_duration_sigma == config.session_duration_sigma
    assert resolved.responsiveness_window == config.responsiveness_window
    assert (
        resolved.temporal_coefficient_scaling
        == config.temporal_coefficient_scaling
    )


def test_temporal_scaling_sensitivity_is_applied() -> None:
    """Verify temporal coefficient scaling changes only that parameter."""

    config = SimulationConfig(
        session_duration_sigma=0.90,
        responsiveness_window=5,
        linucb_alpha=1.0,
        temporal_coefficient_scaling=1.0,
    )

    condition = ExperimentCondition(
        name="TemporalScale-1p2-LinUCB-A5",
        experiment_type="sensitivity",
        algorithm="LinUCB",
        action_space_size=5,
        sensitivity_parameter="temporal_coefficient_scaling",
        sensitivity_value=1.2,
    )

    resolved = _config_for_condition(
        config=config,
        condition=condition,
    )

    assert resolved.temporal_coefficient_scaling == 1.2
    assert resolved.session_duration_sigma == config.session_duration_sigma
    assert resolved.responsiveness_window == config.responsiveness_window
    assert resolved.linucb_alpha == config.linucb_alpha


def test_non_sensitivity_condition_preserves_original_config() -> None:
    """Verify ordinary conditions use the baseline configuration unchanged."""

    config = SimulationConfig()

    condition = ExperimentCondition(
        name="LinUCB-A5",
        experiment_type="primary",
        algorithm="LinUCB",
        action_space_size=5,
    )

    resolved = _config_for_condition(
        config=config,
        condition=condition,
    )

    assert resolved == config

def test_non_stationary_condition_enables_regime_switch() -> None:
    """Non-stationary conditions must enable the environment regime switch."""

    config = _small_config()

    condition = ExperimentCondition(
        name="LinUCB-NonStationary-A5",
        experiment_type="non_stationary",
        algorithm="LinUCB",
        action_space_size=5,
    )

    result = run_experiment_condition(
        config=config,
        condition=condition,
    )

    assert result.condition.experiment_type == "non_stationary"