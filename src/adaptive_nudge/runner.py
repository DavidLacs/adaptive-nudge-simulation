"""Experiment runner for adaptive nudge simulation conditions."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from adaptive_nudge.algorithms import (
    BaseAlgorithm,
    LinUCB,
    Static10MinuteBaseline,
    UCB1,
)
from adaptive_nudge.config import SimulationConfig
from adaptive_nudge.environment import Environment
from adaptive_nudge.evaluation import GroundTruthEvaluator
from adaptive_nudge.experiments import ExperimentCondition
from adaptive_nudge.results import (
    ReplicationMetrics,
    aggregate_replications,
    evaluate_replication,
)
from adaptive_nudge.simulation import (
    DecisionRecord,
    ProgressCallback,
    SimulationRunner,
)


@dataclass(frozen=True)
class ExperimentResult:
    """Results for one experimental condition."""

    condition: ExperimentCondition
    replication_metrics: tuple[ReplicationMetrics, ...]
    aggregated_metrics: dict[str, object]


def _make_algorithm(
    condition: ExperimentCondition,
    config: SimulationConfig,
) -> BaseAlgorithm:
    """Create the algorithm specified by one experiment condition."""

    if condition.algorithm == "LinUCB":
        return LinUCB(
            actions=condition.actions,
            context_dimension=config.context_dimension,
            alpha=config.linucb_alpha,
        )

    if condition.algorithm == "UCB1":
        return UCB1(
            actions=condition.actions,
        )

    if condition.algorithm == "Static10MinuteBaseline":
        return Static10MinuteBaseline(
            actions=condition.actions,
        )

    raise ValueError(
        f"Unsupported algorithm: {condition.algorithm}"
    )


def _config_for_condition(
    config: SimulationConfig,
    condition: ExperimentCondition,
) -> SimulationConfig:
    """Return the simulation configuration for one condition."""

    if getattr(condition, "experiment_type", None) != "sensitivity":
        return config

    if condition.sensitivity_parameter is None:
        raise ValueError(
            "Sensitivity conditions require a sensitivity parameter."
        )

    if condition.sensitivity_value is None:
        raise ValueError(
            "Sensitivity conditions require a sensitivity value."
        )

    return replace(
        config,
        **{
            condition.sensitivity_parameter:
                condition.sensitivity_value
        },
    )


def _group_records_by_replication(
    records: list[DecisionRecord],
) -> list[list[DecisionRecord]]:
    """Group decision records by replication index."""

    if not records:
        raise ValueError(
            "records must contain at least one decision."
        )

    grouped: dict[int, list[DecisionRecord]] = {}

    for record in records:
        grouped.setdefault(
            record.replication,
            [],
        ).append(record)

    actual_replications = sorted(grouped)

    expected_replications = list(
        range(len(grouped))
    )

    if actual_replications != expected_replications:
        raise ValueError(
            "Replication indices must be contiguous "
            "starting at zero."
        )

    return [
        grouped[replication]
        for replication in actual_replications
    ]


def run_experiment_condition(
    config: SimulationConfig,
    condition: ExperimentCondition,
    condition_index: int | None = None,
    total_conditions: int | None = None,
) -> ExperimentResult:
    """Run and evaluate one defined experimental condition."""

    condition_config = _config_for_condition(
        config=config,
        condition=condition,
    )

    algorithm = _make_algorithm(
        condition=condition,
        config=condition_config,
    )

    non_stationary = (
        condition.experiment_type == "non_stationary"
    )

    environment = Environment(
        config=condition_config,
        duration_rng=np.random.default_rng(0),
        context_rng=np.random.default_rng(1),
        non_stationary=non_stationary,
    )

    simulation = SimulationRunner(
        config=condition_config,
        environment=environment,
        algorithm=algorithm,
    )

    if condition_index is not None and total_conditions is not None:
        print(
            f"[{condition_index:02d}/{total_conditions:02d}] "
            f"{condition.name}"
        )

    def report_progress(
        replication: int,
        completed_decisions: int,
        total_decisions: int,
        total_replications: int,
    ) -> None:
        """Print concise live simulation progress."""
        percentage = (
            100.0
            * completed_decisions
            / total_decisions
        )

        if completed_decisions == total_decisions:
            print(
                f"  Replication {replication + 1}/"
                f"{total_replications} complete "
                f"({total_decisions:,}/{total_decisions:,} "
                f"decisions)"
            )
        else:
            print(
                f"  Replication {replication + 1}/"
                f"{total_replications} | "
                f"{completed_decisions:,}/"
                f"{total_decisions:,} decisions "
                f"({percentage:.1f}%)"
            )

    records = simulation.run(
        master_seed=condition_config.base_seed,
        progress_callback=report_progress,
        progress_interval_seconds=5.0,
    )

    print(
        f"  All {len(records):,} decisions simulated."
    )

    records_by_replication = (
        _group_records_by_replication(records)
    )

    evaluator = GroundTruthEvaluator(
        config=condition_config,
        non_stationary=non_stationary,
    )

    evaluated_metrics: list[ReplicationMetrics] = []

    total_replications = len(records_by_replication)

    for replication_index, replication_records in enumerate(
        records_by_replication,
        start=1,
    ):
        metrics = evaluate_replication(
            records=replication_records,
            actions=condition.actions,
            evaluator=evaluator,
            config=condition_config,
            non_stationary=non_stationary,
        )

        evaluated_metrics.append(metrics)

        if (
            replication_index == 1
            or replication_index % 10 == 0
            or replication_index == total_replications
        ):
            print(
                f"  Evaluation: "
                f"{replication_index}/"
                f"{total_replications} complete"
            )

    replication_metrics = tuple(evaluated_metrics)

    aggregated_metrics = aggregate_replications(
        replication_metrics
    )

    if condition_index is not None and total_conditions is not None:
        print(
            f"[{condition_index:02d}/{total_conditions:02d}] "
            f"{condition.name} complete"
        )
        print()

    return ExperimentResult(
        condition=condition,
        replication_metrics=replication_metrics,
        aggregated_metrics=aggregated_metrics,
    )


def run_experiment_suite(
    config: SimulationConfig,
    conditions: tuple[ExperimentCondition, ...],
) -> tuple[ExperimentResult, ...]:
    """Run all supplied experimental conditions."""

    if not conditions:
        raise ValueError(
            "conditions must contain at least one experiment condition."
        )

    results: list[ExperimentResult] = []

    total_conditions = len(conditions)

    for condition_index, condition in enumerate(
        conditions,
        start=1,
    ):
        results.append(
            run_experiment_condition(
                config=config,
                condition=condition,
                condition_index=condition_index,
                total_conditions=total_conditions,
            )
        )

    return tuple(results)