"""Tests for simulation result evaluation and aggregation."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from adaptive_nudge.algorithms import (
    LinUCB,
    Static10MinuteBaseline,
)
from adaptive_nudge.config import (
    ACTION_SPACES,
    SimulationConfig,
)
from adaptive_nudge.environment import Environment
from adaptive_nudge.evaluation import GroundTruthEvaluator
from adaptive_nudge.results import (
    AggregateMetric,
    ReplicationMetrics,
    aggregate_metric,
    aggregate_replications,
    evaluate_replication,
)
from adaptive_nudge.simulation import SimulationRunner


def test_aggregate_metric_known_values() -> None:
    """Verify mean, sample SD, and 95% t-based CI."""

    values = (
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    )

    result = aggregate_metric(values)

    expected_mean = 3.0

    expected_sd = math.sqrt(2.5)

    expected_se = (
        expected_sd
        / math.sqrt(5)
    )

    expected_t = 2.7764451051977987

    expected_margin = (
        expected_t
        * expected_se
    )

    assert np.isclose(
        result.mean,
        expected_mean,
    )

    assert np.isclose(
        result.standard_deviation,
        expected_sd,
    )

    assert result.num_replications == 5

    assert np.isclose(
        result.confidence_interval_95[0],
        expected_mean - expected_margin,
    )

    assert np.isclose(
        result.confidence_interval_95[1],
        expected_mean + expected_margin,
    )


def test_single_replication_has_zero_uncertainty() -> None:
    """Verify one replication has zero estimated uncertainty."""

    result = aggregate_metric(
        (0.25,)
    )

    assert result.mean == 0.25
    assert result.standard_deviation == 0.0
    assert result.confidence_interval_95 == (
        0.25,
        0.25,
    )
    assert result.num_replications == 1


def test_empty_values_are_rejected() -> None:
    """Verify empty aggregation input is rejected."""

    with pytest.raises(ValueError):
        aggregate_metric(())


def test_nonfinite_values_are_rejected() -> None:
    """Verify non-finite aggregation input is rejected."""

    with pytest.raises(ValueError):
        aggregate_metric(
            (
                1.0,
                np.nan,
            )
        )


def test_replication_level_statistics_are_not_pooled() -> None:
    """Verify aggregation operates on replication-level estimates."""

    replication_means = (
        0.2,
        0.4,
        0.6,
    )

    result = aggregate_metric(
        replication_means
    )

    assert np.isclose(
        result.mean,
        0.4,
    )

    assert np.isclose(
        result.standard_deviation,
        0.2,
    )

    assert result.num_replications == 3


def test_evaluate_replication_from_actual_simulation() -> None:
    """Verify evaluation against one real stationary simulation replication."""

    base_config = SimulationConfig()

    config = replace(
        base_config,
        decisions_per_replication=1_000,
        num_replications=1,
    )

    environment = Environment(
        config=config,
        duration_rng=np.random.default_rng(0),
        context_rng=np.random.default_rng(1),
    )

    algorithm = LinUCB(
        actions=ACTION_SPACES[5],
        context_dimension=config.context_dimension,
        alpha=config.linucb_alpha,
    )

    runner = SimulationRunner(
        config=config,
        environment=environment,
        algorithm=algorithm,
    )

    records = runner.run_replication(
        replication=0,
        master_seed=config.base_seed,
    )

    evaluator = GroundTruthEvaluator(
        config=config,
        non_stationary=False,
    )

    metrics = evaluate_replication(
        records=records,
        actions=ACTION_SPACES[5],
        evaluator=evaluator,
        config=config,
        non_stationary=False,
        convergence_window=100,
    )

    assert isinstance(
        metrics,
        ReplicationMetrics,
    )

    assert metrics.replication == 0

    assert metrics.algorithm == "LinUCB"

    assert metrics.num_decisions == 1_000

    assert (
        metrics.cumulative_pseudo_regret
        >= 0.0
    )

    assert (
        metrics.mean_pseudo_regret
        >= 0.0
    )

    assert (
        0.0
        <= metrics.successful_session_exit_rate
        <= 1.0
    )

    assert (
        0.0
        <= metrics.optimal_arm_selection_rate
        <= 1.0
    )

    # Regime-specific metrics are undefined for stationary experiments.
    assert (
        metrics.pre_change_optimal_arm_selection_rate
        is None
    )

    assert (
        metrics.post_change_optimal_arm_selection_rate
        is None
    )

    assert (
        metrics.convergence_index is None
        or (
            99
            <= metrics.convergence_index
            < 1_000
        )
    )

    assert metrics.recovery_index is None

    assert metrics.recovery_delay is None


def test_aggregate_replications_known_metrics() -> None:
    """Verify across-replication aggregation."""

    metrics = (
        ReplicationMetrics(
            replication=0,
            algorithm="LinUCB",
            num_decisions=1_000,
            cumulative_pseudo_regret=10.0,
            mean_pseudo_regret=0.010,
            successful_session_exit_rate=0.20,
            optimal_arm_selection_rate=0.50,
            pre_change_optimal_arm_selection_rate=0.60,
            post_change_optimal_arm_selection_rate=0.40,
            convergence_index=900,
            recovery_index=800,
            recovery_delay=300,
        ),
        ReplicationMetrics(
            replication=1,
            algorithm="LinUCB",
            num_decisions=1_000,
            cumulative_pseudo_regret=20.0,
            mean_pseudo_regret=0.020,
            successful_session_exit_rate=0.40,
            optimal_arm_selection_rate=0.70,
            pre_change_optimal_arm_selection_rate=0.70,
            post_change_optimal_arm_selection_rate=0.60,
            convergence_index=950,
            recovery_index=850,
            recovery_delay=350,
        ),
        ReplicationMetrics(
            replication=2,
            algorithm="LinUCB",
            num_decisions=1_000,
            cumulative_pseudo_regret=30.0,
            mean_pseudo_regret=0.030,
            successful_session_exit_rate=0.60,
            optimal_arm_selection_rate=0.90,
            pre_change_optimal_arm_selection_rate=0.80,
            post_change_optimal_arm_selection_rate=0.70,
            convergence_index=980,
            recovery_index=900,
            recovery_delay=400,
        ),
    )

    aggregated = aggregate_replications(
        metrics
    )

    assert set(aggregated) == {
        "cumulative_pseudo_regret",
        "mean_pseudo_regret",
        "successful_session_exit_rate",
        "optimal_arm_selection_rate",
        "pre_change_optimal_arm_selection_rate",
        "post_change_optimal_arm_selection_rate",
        "convergence_index",
        "recovery_index",
        "recovery_delay",
    }

    assert np.isclose(
        aggregated[
            "cumulative_pseudo_regret"
        ].mean,
        20.0,
    )

    assert np.isclose(
        aggregated[
            "mean_pseudo_regret"
        ].mean,
        0.020,
    )

    assert np.isclose(
        aggregated[
            "successful_session_exit_rate"
        ].mean,
        0.40,
    )

    assert np.isclose(
        aggregated[
            "optimal_arm_selection_rate"
        ].mean,
        0.70,
    )

    assert np.isclose(
        aggregated[
            "pre_change_optimal_arm_selection_rate"
        ].mean,
        0.70,
    )

    assert np.isclose(
        aggregated[
            "post_change_optimal_arm_selection_rate"
        ].mean,
        0.5666666666666667,
    )

    assert np.isclose(
        aggregated[
            "convergence_index"
        ].mean,
        943.3333333333334,
    )

    assert np.isclose(
        aggregated[
            "recovery_index"
        ].mean,
        850.0,
    )

    assert np.isclose(
        aggregated[
            "recovery_delay"
        ].mean,
        350.0,
    )

    assert all(
        isinstance(
            value,
            AggregateMetric,
        )
        for value in aggregated.values()
    )

    assert all(
        value.num_replications == 3
        for value in aggregated.values()
    )


def test_undefined_recovery_is_not_treated_as_zero() -> None:
    """Verify unsuccessful recovery is excluded from timing statistics."""

    metrics = (
        ReplicationMetrics(
            replication=0,
            algorithm="LinUCB",
            num_decisions=1_000,
            cumulative_pseudo_regret=10.0,
            mean_pseudo_regret=0.010,
            successful_session_exit_rate=0.20,
            optimal_arm_selection_rate=0.50,
            pre_change_optimal_arm_selection_rate=0.60,
            post_change_optimal_arm_selection_rate=0.40,
            convergence_index=None,
            recovery_index=None,
            recovery_delay=None,
        ),
        ReplicationMetrics(
            replication=1,
            algorithm="LinUCB",
            num_decisions=1_000,
            cumulative_pseudo_regret=20.0,
            mean_pseudo_regret=0.020,
            successful_session_exit_rate=0.40,
            optimal_arm_selection_rate=0.70,
            pre_change_optimal_arm_selection_rate=0.70,
            post_change_optimal_arm_selection_rate=0.60,
            convergence_index=900,
            recovery_index=800,
            recovery_delay=300,
        ),
    )

    aggregated = aggregate_replications(
        metrics
    )

    assert np.isclose(
        aggregated["convergence_index"].mean,
        900.0,
    )

    assert np.isclose(
        aggregated["recovery_index"].mean,
        800.0,
    )

    assert np.isclose(
        aggregated["recovery_delay"].mean,
        300.0,
    )


def test_mixed_algorithms_are_rejected() -> None:
    """Verify one aggregation call cannot mix algorithms."""

    metrics = (
        ReplicationMetrics(
            replication=0,
            algorithm="LinUCB",
            num_decisions=1_000,
            cumulative_pseudo_regret=10.0,
            mean_pseudo_regret=0.010,
            successful_session_exit_rate=0.20,
            optimal_arm_selection_rate=0.50,
            pre_change_optimal_arm_selection_rate=0.60,
            post_change_optimal_arm_selection_rate=0.40,
            convergence_index=None,
            recovery_index=None,
            recovery_delay=None,
        ),
        ReplicationMetrics(
            replication=0,
            algorithm="UCB1",
            num_decisions=1_000,
            cumulative_pseudo_regret=20.0,
            mean_pseudo_regret=0.020,
            successful_session_exit_rate=0.30,
            optimal_arm_selection_rate=0.60,
            pre_change_optimal_arm_selection_rate=0.50,
            post_change_optimal_arm_selection_rate=0.30,
            convergence_index=None,
            recovery_index=None,
            recovery_delay=None,
        ),
    )

    with pytest.raises(ValueError):
        aggregate_replications(metrics)


def test_static_baseline_has_no_adaptive_learning_or_regime_metrics() -> None:
    """Verify the static baseline has no adaptive-learning metrics."""

    base_config = SimulationConfig()

    config = replace(
        base_config,
        decisions_per_replication=1_000,
        num_replications=1,
    )

    environment = Environment(
        config=config,
        duration_rng=np.random.default_rng(0),
        context_rng=np.random.default_rng(1),
    )

    algorithm = Static10MinuteBaseline(
        actions=ACTION_SPACES[5],
    )

    runner = SimulationRunner(
        config=config,
        environment=environment,
        algorithm=algorithm,
    )

    records = runner.run_replication(
        replication=0,
        master_seed=config.base_seed,
    )

    evaluator = GroundTruthEvaluator(
        config=config,
        non_stationary=False,
    )

    metrics = evaluate_replication(
        records=records,
        actions=ACTION_SPACES[5],
        evaluator=evaluator,
        config=config,
        non_stationary=False,
        convergence_window=100,
    )

    assert isinstance(
        metrics,
        ReplicationMetrics,
    )

    assert metrics.algorithm == "Static10MinuteBaseline"

    assert metrics.num_decisions == 1_000

    assert (
        metrics.cumulative_pseudo_regret
        >= 0.0
    )

    assert (
        metrics.mean_pseudo_regret
        >= 0.0
    )

    assert (
        0.0
        <= metrics.successful_session_exit_rate
        <= 1.0
    )

    assert (
        0.0
        <= metrics.optimal_arm_selection_rate
        <= 1.0
    )

    # Regime-specific metrics are undefined for stationary experiments.
    assert (
        metrics.pre_change_optimal_arm_selection_rate
        is None
    )

    assert (
        metrics.post_change_optimal_arm_selection_rate
        is None
    )

    assert metrics.convergence_index is None
    assert metrics.recovery_index is None
    assert metrics.recovery_delay is None