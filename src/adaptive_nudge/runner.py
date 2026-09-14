"""Experiment runner for adaptive nudge simulation conditions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive_nudge.algorithms import (
    BaseAlgorithm,
    LinUCB,
    Static10MinuteBaseline,
    UCB1,
)
from adaptive_nudge.config import ACTION_SPACES, SimulationConfig
from adaptive_nudge.environment import Environment
from adaptive_nudge.evaluation import GroundTruthEvaluator
from adaptive_nudge.results import (
    ReplicationMetrics,
    aggregate_replications,
    evaluate_replication,
)
from adaptive_nudge.simulation import DecisionRecord, SimulationRunner


@dataclass(frozen=True)
class ExperimentResult:
    """Results for one algorithm/action-space condition."""

    algorithm: str
    action_space_size: int
    actions: tuple[int, ...]
    replication_metrics: tuple[ReplicationMetrics, ...]
    aggregated_metrics: dict[str, object]


def _make_algorithm(
    algorithm_name: str,
    actions: tuple[int, ...],
    config: SimulationConfig,
) -> BaseAlgorithm:
    """Create the requested algorithm for one condition."""

    if algorithm_name == "LinUCB":
        return LinUCB(
            actions=actions,
            context_dimension=config.context_dimension,
            alpha=config.linucb_alpha,
        )

    if algorithm_name == "UCB1":
        return UCB1(
            actions=actions,
        )

    if algorithm_name == "Static10MinuteBaseline":
        return Static10MinuteBaseline(
            actions=actions,
        )

    raise ValueError(
        f"Unsupported algorithm: {algorithm_name}"
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

    actual_replications = sorted(
        grouped
    )

    expected_replications = list(
        range(
            len(grouped)
        )
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
    algorithm_name: str,
    action_space_size: int,
) -> ExperimentResult:
    """Run and evaluate one algorithm/action-space condition."""

    if action_space_size not in ACTION_SPACES:
        raise ValueError(
            f"Unsupported action-space size: {action_space_size}"
        )

    actions = tuple(
        ACTION_SPACES[action_space_size]
    )

    algorithm = _make_algorithm(
        algorithm_name=algorithm_name,
        actions=actions,
        config=config,
    )

    environment = Environment(
        config=config,
        duration_rng=np.random.default_rng(0),
        context_rng=np.random.default_rng(1),
    )

    simulation = SimulationRunner(
        config=config,
        environment=environment,
        algorithm=algorithm,
    )

    records = simulation.run(
        master_seed=config.base_seed
    )

    records_by_replication = (
        _group_records_by_replication(records)
    )

    evaluator = GroundTruthEvaluator(
        config=config,
    )

    replication_metrics = tuple(
        evaluate_replication(
            records=replication_records,
            actions=actions,
            evaluator=evaluator,
            config=config,
        )
        for replication_records in records_by_replication
    )

    aggregated_metrics = aggregate_replications(
        replication_metrics
    )

    return ExperimentResult(
        algorithm=algorithm_name,
        action_space_size=action_space_size,
        actions=actions,
        replication_metrics=replication_metrics,
        aggregated_metrics=aggregated_metrics,
    )