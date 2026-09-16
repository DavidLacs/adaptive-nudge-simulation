"""Tests for the adaptive nudge simulation runner."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_nudge.algorithms import LinUCB, UCB1
from adaptive_nudge.config import SimulationConfig
from adaptive_nudge.environment import Environment
from adaptive_nudge.simulation import SimulationRunner


def make_runner(
    config: SimulationConfig | None = None,
) -> SimulationRunner:
    """Create a deterministic simulation runner for testing."""
    if config is None:
        config = SimulationConfig(
            decisions_per_replication=10,
            num_replications=2,
        )

    environment = Environment(
        config=config,
        duration_rng=np.random.default_rng(0),
        context_rng=np.random.default_rng(1),
    )

    algorithm = LinUCB(
        actions=(3, 7, 10, 15, 20),
        context_dimension=config.context_dimension,
        alpha=config.linucb_alpha,
    )

    return SimulationRunner(
        config=config,
        environment=environment,
        algorithm=algorithm,
    )


def test_run_replication_returns_expected_number_of_records() -> None:
    """Each replication must produce one record per decision."""
    config = SimulationConfig(
        decisions_per_replication=10,
        num_replications=2,
    )
    runner = make_runner(config)

    records = runner.run_replication(
        replication=0,
        master_seed=config.base_seed,
    )

    assert len(records) == 10


def test_run_returns_all_configured_replications() -> None:
    """The full run must produce all configured replication records."""
    config = SimulationConfig(
        decisions_per_replication=10,
        num_replications=2,
    )
    runner = make_runner(config)

    records = runner.run(
        master_seed=config.base_seed,
    )

    assert len(records) == 20
    assert {
        record.replication
        for record in records
    } == {0, 1}


def test_same_master_seed_reproduces_identical_records() -> None:
    """Repeated runs with the same seed must be reproducible."""
    config = SimulationConfig(
        decisions_per_replication=10,
        num_replications=2,
    )

    first_runner = make_runner(config)
    second_runner = make_runner(config)

    first = first_runner.run(
        master_seed=config.base_seed,
    )
    second = second_runner.run(
        master_seed=config.base_seed,
    )

    assert len(first) == len(second)

    for first_record, second_record in zip(
        first,
        second,
    ):
        assert first_record.replication == second_record.replication
        assert first_record.master_seed == second_record.master_seed
        assert first_record.replication_seed == second_record.replication_seed
        assert first_record.algorithm == second_record.algorithm
        assert first_record.duration_stream == second_record.duration_stream
        assert first_record.context_stream == second_record.context_stream
        assert first_record.response_stream == second_record.response_stream
        assert first_record.decision_index == second_record.decision_index
        assert first_record.timing_minutes == second_record.timing_minutes
        assert first_record.duration_minutes == second_record.duration_minutes
        assert first_record.time_of_day == second_record.time_of_day
        assert first_record.recent_responsiveness == second_record.recent_responsiveness
        assert np.array_equal(
            first_record.context,
            second_record.context,
        )
        assert first_record.opportunity == second_record.opportunity
        assert first_record.response == second_record.response
        assert first_record.reward == second_record.reward


def test_different_replications_receive_different_replication_seeds() -> None:
    """Configured replications must have distinct derived seeds."""
    config = SimulationConfig(
        decisions_per_replication=5,
        num_replications=3,
    )
    runner = make_runner(config)

    records = runner.run(
        master_seed=config.base_seed,
    )

    replication_seeds = {
        record.replication: record.replication_seed
        for record in records
    }

    assert len(replication_seeds) == 3
    assert len(set(replication_seeds.values())) == 3


def test_record_replication_metadata_is_consistent() -> None:
    """Every record must contain metadata matching its replication."""
    config = SimulationConfig(
        decisions_per_replication=5,
        num_replications=2,
    )
    runner = make_runner(config)

    records = runner.run(
        master_seed=config.base_seed,
    )

    for record in records:
        assert record.master_seed == config.base_seed
        assert record.replication in {0, 1}
        assert 0 <= record.decision_index < 5
        assert record.algorithm == "LinUCB"


def test_record_action_belongs_to_algorithm_action_space() -> None:
    """Selected actions must belong to the configured LinUCB action space."""
    config = SimulationConfig(
        decisions_per_replication=10,
        num_replications=1,
    )
    runner = make_runner(config)

    records = runner.run(
        master_seed=config.base_seed,
    )

    valid_actions = {3, 7, 10, 15, 20}

    assert all(
        record.timing_minutes in valid_actions
        for record in records
    )


def test_negative_master_seed_is_rejected() -> None:
    """Negative master seeds must fail explicitly."""
    runner = make_runner()

    with pytest.raises(ValueError):
        runner.run(master_seed=-1)


def test_invalid_replication_is_rejected() -> None:
    """Replication indices outside the configured range must fail."""
    config = SimulationConfig(
        decisions_per_replication=5,
        num_replications=2,
    )
    runner = make_runner(config)

    with pytest.raises(ValueError):
        runner.run_replication(
            replication=2,
            master_seed=config.base_seed,
        )


def test_negative_replication_is_rejected() -> None:
    """Negative replication indices must fail explicitly."""
    runner = make_runner()

    with pytest.raises(ValueError):
        runner.run_replication(
            replication=-1,
            master_seed=20260903,
        )


def test_response_stream_is_algorithm_specific() -> None:
    """Different algorithms must receive distinct response-stream identifiers."""
    config = SimulationConfig(
        decisions_per_replication=5,
        num_replications=1,
    )

    linucb_runner = make_runner(config)

    linucb_records = linucb_runner.run(
        master_seed=config.base_seed,
    )

    assert all(
        record.response_stream
        == "PCG64:master/replication/response/LinUCB"
        for record in linucb_records
    )

def test_algorithms_share_exogenous_duration_and_time_of_day_sequences() -> None:
    """Different algorithms must receive the same exogenous session sequence."""
    config = SimulationConfig(
        decisions_per_replication=10,
        num_replications=1,
    )

    linucb_runner = make_runner(config)

    ucb1_algorithm = UCB1(
        actions=(3, 7, 10, 15, 20),
    )

    ucb1_environment = Environment(
        config=config,
        duration_rng=np.random.default_rng(0),
        context_rng=np.random.default_rng(1),
    )

    ucb1_runner = SimulationRunner(
        config=config,
        environment=ucb1_environment,
        algorithm=ucb1_algorithm,
    )

    linucb_records = linucb_runner.run(
        master_seed=config.base_seed,
    )

    ucb1_records = ucb1_runner.run(
        master_seed=config.base_seed,
    )

    assert len(linucb_records) == len(ucb1_records)

    for linucb_record, ucb1_record in zip(
        linucb_records,
        ucb1_records,
    ):
        assert (
            linucb_record.replication
            == ucb1_record.replication
        )

        assert (
            linucb_record.decision_index
            == ucb1_record.decision_index
        )

        assert (
            linucb_record.duration_minutes
            == ucb1_record.duration_minutes
        )

        assert (
            linucb_record.time_of_day
            == ucb1_record.time_of_day
        )

        assert np.array_equal(
            linucb_record.context[:4],
            ucb1_record.context[:4],
        )