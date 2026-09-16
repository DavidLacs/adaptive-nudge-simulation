"""Replication-level evaluation and statistical aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import t

from adaptive_nudge.config import SimulationConfig
from adaptive_nudge.evaluation import (
    GroundTruthEvaluator,
    convergence_index,
    recovery_delay,
    recovery_index,
)
from adaptive_nudge.simulation import DecisionRecord


@dataclass(frozen=True)
class ReplicationMetrics:
    """Performance metrics calculated for one simulation replication."""

    replication: int
    algorithm: str
    num_decisions: int

    cumulative_pseudo_regret: float
    mean_pseudo_regret: float
    successful_session_exit_rate: float
    optimal_arm_selection_rate: float

    pre_change_optimal_arm_selection_rate: float | None
    post_change_optimal_arm_selection_rate: float | None

    convergence_index: int | None
    recovery_index: int | None
    recovery_delay: int | None


@dataclass(frozen=True)
class AggregateMetric:
    """Across-replication summary for one performance metric."""

    mean: float
    standard_deviation: float
    confidence_interval_95: tuple[float, float]
    num_replications: int


def _mean_indicator_rate(
    indicators: np.ndarray,
    start: int,
    end: int,
) -> float:
    """Return the mean optimal-arm indicator over [start, end)."""

    if indicators.ndim != 1:
        raise ValueError(
            "indicators must be one-dimensional."
        )

    if not 0 <= start <= end <= len(indicators):
        raise ValueError(
            "indicator range must satisfy "
            "0 <= start <= end <= len(indicators)."
        )

    if start == end:
        raise ValueError(
            "indicator range must contain at least one decision."
        )

    return float(
        np.mean(
            indicators[start:end]
        )
    )


def evaluate_replication(
    records: Sequence[DecisionRecord],
    actions: Sequence[int],
    evaluator: GroundTruthEvaluator,
    config: SimulationConfig,
    non_stationary: bool = False,
    convergence_window: int | None = None,
    convergence_threshold: float | None = None,
) -> ReplicationMetrics:
    """Calculate evaluation metrics for one complete replication."""

    if not records:
        raise ValueError(
            "records must contain at least one decision."
        )

    first_record = records[0]

    if any(
        record.replication != first_record.replication
        for record in records
    ):
        raise ValueError(
            "All records must belong to the same replication."
        )

    if any(
        record.algorithm != first_record.algorithm
        for record in records
    ):
        raise ValueError(
            "All records must belong to the same algorithm."
        )

    decision_indices = [
        record.decision_index
        for record in records
    ]

    expected_indices = list(
        range(len(records))
    )

    if decision_indices != expected_indices:
        raise ValueError(
            "records must contain one contiguous decision sequence "
            "starting at decision index 0."
        )

    available_actions = tuple(
        sorted(
            set(actions)
        )
    )

    if not available_actions:
        raise ValueError(
            "actions must contain at least one timing."
        )

    if any(
        record.timing_minutes not in available_actions
        for record in records
    ):
        raise ValueError(
            "Every selected action must belong to actions."
        )

    indicators = np.empty(
        len(records),
        dtype=np.int8,
    )

    pseudo_regrets = np.empty(
        len(records),
        dtype=float,
    )

    for index, record in enumerate(records):
        expected_rewards = np.array(
            [
                evaluator.expected_reward(
                    timing_minutes=timing,
                    context=record.context,
                    decision_index=record.decision_index,
                )
                for timing in available_actions
            ],
            dtype=float,
        )

        optimal_reward = float(
            np.max(expected_rewards)
        )

        selected_action_index = available_actions.index(
            record.timing_minutes
        )

        selected_reward = float(
            expected_rewards[selected_action_index]
        )

        indicators[index] = int(
            math.isclose(
                selected_reward,
                optimal_reward,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )

        pseudo_regrets[index] = max(
            0.0,
            optimal_reward - selected_reward,
        )

    cumulative_pseudo_regret = float(
        np.sum(pseudo_regrets)
    )

    mean_pseudo_regret = float(
        np.mean(pseudo_regrets)
    )

    successful_session_exit_rate = float(
        np.mean(
            [
                record.reward
                for record in records
            ]
        )
    )

    optimal_arm_selection_rate = float(
        np.mean(indicators)
    )

    pre_change_optimal_arm_selection_rate: float | None = None
    post_change_optimal_arm_selection_rate: float | None = None

    if non_stationary:
        regime_change_point = config.regime_change_point

        if not 0 < regime_change_point < len(records):
            raise ValueError(
                "regime_change_point must divide the replication "
                "into two non-empty regimes."
            )

        pre_change_optimal_arm_selection_rate = (
            _mean_indicator_rate(
                indicators=indicators,
                start=0,
                end=regime_change_point,
            )
        )

        post_change_optimal_arm_selection_rate = (
            _mean_indicator_rate(
                indicators=indicators,
                start=regime_change_point,
                end=len(records),
            )
        )

    if convergence_window is None:
        convergence_window = config.convergence_window

    if convergence_threshold is None:
        convergence_threshold = (
            config.convergence_threshold
        )

    if first_record.algorithm == "Static10MinuteBaseline":
        convergence = None
        recovery = None
        delay = None
    else:
        convergence = convergence_index(
            indicators=indicators,
            window=convergence_window,
            threshold=convergence_threshold,
        )

        if non_stationary:
            recovery = recovery_index(
                indicators=indicators,
                regime_change_point=config.regime_change_point,
                window=convergence_window,
                threshold=convergence_threshold,
            )

            delay = recovery_delay(
                recovery_index_value=recovery,
                regime_change_point=config.regime_change_point,
            )
        else:
            recovery = None
            delay = None

    return ReplicationMetrics(
        replication=first_record.replication,
        algorithm=first_record.algorithm,
        num_decisions=len(records),
        cumulative_pseudo_regret=(
            cumulative_pseudo_regret
        ),
        mean_pseudo_regret=(
            mean_pseudo_regret
        ),
        successful_session_exit_rate=(
            successful_session_exit_rate
        ),
        optimal_arm_selection_rate=(
            optimal_arm_selection_rate
        ),
        pre_change_optimal_arm_selection_rate=(
            pre_change_optimal_arm_selection_rate
        ),
        post_change_optimal_arm_selection_rate=(
            post_change_optimal_arm_selection_rate
        ),
        convergence_index=convergence,
        recovery_index=recovery,
        recovery_delay=delay,
    )


def aggregate_metric(
    values: Sequence[float],
) -> AggregateMetric:
    """Aggregate replication-level values with a 95% CI."""

    if not values:
        raise ValueError(
            "values must contain at least one observation."
        )

    array = np.asarray(
        values,
        dtype=float,
    )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            "values must contain only finite numbers."
        )

    num_replications = len(array)

    mean = float(
        np.mean(array)
    )

    if num_replications == 1:
        standard_deviation = 0.0
        margin_of_error = 0.0

    else:
        standard_deviation = float(
            np.std(
                array,
                ddof=1,
            )
        )

        standard_error = (
            standard_deviation
            / math.sqrt(num_replications)
        )

        critical_value = float(
            t.ppf(
                0.975,
                df=num_replications - 1,
            )
        )

        margin_of_error = (
            critical_value
            * standard_error
        )

    confidence_interval = (
        mean - margin_of_error,
        mean + margin_of_error,
    )

    return AggregateMetric(
        mean=mean,
        standard_deviation=standard_deviation,
        confidence_interval_95=confidence_interval,
        num_replications=num_replications,
    )


def aggregate_replications(
    metrics: Sequence[ReplicationMetrics],
) -> dict[str, AggregateMetric]:
    """Aggregate replication-level metrics across replications."""

    if not metrics:
        raise ValueError(
            "metrics must contain at least one replication."
        )

    algorithms = {
        metric.algorithm
        for metric in metrics
    }

    if len(algorithms) != 1:
        raise ValueError(
            "All metrics must belong to the same algorithm."
        )

    metric_values = {
        "cumulative_pseudo_regret": [
            metric.cumulative_pseudo_regret
            for metric in metrics
        ],
        "mean_pseudo_regret": [
            metric.mean_pseudo_regret
            for metric in metrics
        ],
        "successful_session_exit_rate": [
            metric.successful_session_exit_rate
            for metric in metrics
        ],
        "optimal_arm_selection_rate": [
            metric.optimal_arm_selection_rate
            for metric in metrics
        ],
    }

    aggregated = {
        name: aggregate_metric(values)
        for name, values in metric_values.items()
    }

    pre_change_values = [
        metric.pre_change_optimal_arm_selection_rate
        for metric in metrics
        if metric.pre_change_optimal_arm_selection_rate is not None
    ]

    post_change_values = [
        metric.post_change_optimal_arm_selection_rate
        for metric in metrics
        if metric.post_change_optimal_arm_selection_rate is not None
    ]

    if pre_change_values:
        aggregated["pre_change_optimal_arm_selection_rate"] = (
            aggregate_metric(
                pre_change_values
            )
        )

    if post_change_values:
        aggregated["post_change_optimal_arm_selection_rate"] = (
            aggregate_metric(
                post_change_values
            )
        )

    convergence_values = [
        metric.convergence_index
        for metric in metrics
        if metric.convergence_index is not None
    ]

    recovery_values = [
        metric.recovery_index
        for metric in metrics
        if metric.recovery_index is not None
    ]

    recovery_delay_values = [
        metric.recovery_delay
        for metric in metrics
        if metric.recovery_delay is not None
    ]

    if convergence_values:
        aggregated["convergence_index"] = (
            aggregate_metric(
                convergence_values
            )
        )

    if recovery_values:
        aggregated["recovery_index"] = (
            aggregate_metric(
                recovery_values
            )
        )

    if recovery_delay_values:
        aggregated["recovery_delay"] = (
            aggregate_metric(
                recovery_delay_values
            )
        )

    return aggregated