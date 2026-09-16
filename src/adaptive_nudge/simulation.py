"""Simulation runner for adaptive nudge-timing experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from collections.abc import Callable

import numpy as np

from .algorithms import BaseAlgorithm
from .config import SimulationConfig
from .environment import Environment, SessionState


@dataclass(frozen=True)
class DecisionRecord:
    """Record of one session-level simulation decision."""

    replication: int
    master_seed: int
    replication_seed: int
    algorithm: str
    duration_stream: str
    context_stream: str
    response_stream: str
    decision_index: int
    timing_minutes: int
    duration_minutes: float
    time_of_day: str
    recent_responsiveness: float
    context: np.ndarray
    opportunity: int
    response: int
    reward: float


ProgressCallback = Callable[[int, int, int, int], None]


class SimulationRunner:
    """Run one algorithm under one simulation configuration."""

    _DURATION_STREAM = 0
    _CONTEXT_STREAM = 1
    _RESPONSE_STREAM = 2

    def __init__(
        self,
        config: SimulationConfig,
        environment: Environment,
        algorithm: BaseAlgorithm,
    ) -> None:
        self.config = config
        self.environment = environment
        self.algorithm = algorithm

    @staticmethod
    def _algorithm_stream_id(
        algorithm: BaseAlgorithm,
    ) -> int:
        """Create a stable stream identifier for an algorithm.

        The identifier is derived from the algorithm class name using
        SHA-256 so that adding another algorithm does not require
        modifying this simulation runner.
        """
        algorithm_name = algorithm.__class__.__name__

        digest = hashlib.sha256(
            algorithm_name.encode("utf-8")
        ).digest()

        return int.from_bytes(
            digest[:8],
            byteorder="little",
            signed=False,
        )

    def _make_rngs(
        self,
        master_seed: int,
        replication: int,
    ) -> tuple[
        np.random.Generator,
        np.random.Generator,
        np.random.Generator,
        int,
    ]:
        """Create reproducible RNG streams for one replication."""
        if master_seed < 0:
            raise ValueError(
                "master_seed must be non-negative."
            )

        if replication < 0:
            raise ValueError(
                "replication must be non-negative."
            )

        if replication >= self.config.num_replications:
            raise ValueError(
                "replication must be smaller than "
                "config.num_replications."
            )

        master_sequence = np.random.SeedSequence(
            master_seed
        )

        replication_sequences = master_sequence.spawn(
            self.config.num_replications
        )

        replication_sequence = (
            replication_sequences[replication]
        )

        replication_seed = int(
            replication_sequence.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )

        duration_sequence = (
            replication_sequence.spawn(3)
        )

        duration_rng = np.random.default_rng(
            duration_sequence[self._DURATION_STREAM]
        )

        context_rng = np.random.default_rng(
            duration_sequence[self._CONTEXT_STREAM]
        )

        algorithm_stream_id = (
            self._algorithm_stream_id(self.algorithm)
        )

        response_sequence = np.random.SeedSequence(
            [
                master_seed,
                replication,
                self._RESPONSE_STREAM,
                algorithm_stream_id,
            ]
        )

        response_rng = np.random.default_rng(
            response_sequence
        )

        return (
            duration_rng,
            context_rng,
            response_rng,
            replication_seed,
        )

    def run_replication(
        self,
        replication: int,
        master_seed: int,
        progress_callback: ProgressCallback | None = None,
        progress_interval_seconds: float = 5.0,
    ) -> list[DecisionRecord]:
        """Run one independent replication.

        If a progress callback is supplied, progress is reported
        approximately every ``progress_interval_seconds`` and once
        when the replication finishes.
        """
        if replication < 0:
            raise ValueError(
                "replication must be non-negative."
            )

        if replication >= self.config.num_replications:
            raise ValueError(
                "replication must be smaller than "
                "config.num_replications."
            )

        if progress_interval_seconds <= 0.0:
            raise ValueError(
                "progress_interval_seconds must be positive."
            )

        (
            duration_rng,
            context_rng,
            response_rng,
            replication_seed,
        ) = self._make_rngs(
            master_seed=master_seed,
            replication=replication,
        )

        self.environment.set_rngs(
            duration_rng=duration_rng,
            context_rng=context_rng,
        )

        self.environment.reset()
        self.algorithm.reset()

        algorithm_name = (
            self.algorithm.__class__.__name__
        )

        duration_stream = (
            "PCG64:master/replication/duration"
        )

        context_stream = (
            "PCG64:master/replication/context"
        )

        response_stream = (
            "PCG64:master/replication/"
            f"response/{algorithm_name}"
        )

        records: list[DecisionRecord] = []

        start_time = time.monotonic()
        next_progress_time = (
            start_time + progress_interval_seconds
        )

        total_decisions = (
            self.config.decisions_per_replication
        )

        for decision_index in range(
            total_decisions
        ):
            session: SessionState = (
                self.environment.generate_session()
            )

            action = self.algorithm.select_action(
                session.context
            )

            (
                opportunity,
                response,
                reward,
            ) = self.environment.evaluate_action(
                session=session,
                timing_minutes=action,
                response_rng=response_rng,
                decision_index=decision_index,
            )

            self.algorithm.update(
                action=action,
                context=session.context,
                reward=reward,
            )

            records.append(
                DecisionRecord(
                    replication=replication,
                    master_seed=master_seed,
                    replication_seed=replication_seed,
                    algorithm=algorithm_name,
                    duration_stream=duration_stream,
                    context_stream=context_stream,
                    response_stream=response_stream,
                    decision_index=decision_index,
                    timing_minutes=action,
                    duration_minutes=session.duration_minutes,
                    time_of_day=session.time_of_day,
                    recent_responsiveness=(
                        session.recent_responsiveness
                    ),
                    context=session.context.copy(),
                    opportunity=opportunity,
                    response=response,
                    reward=reward,
                )
            )

            if progress_callback is not None:
                current_time = time.monotonic()

                if current_time >= next_progress_time:
                    progress_callback(
                        replication,
                        decision_index + 1,
                        total_decisions,
                        self.config.num_replications,
                    )

                    next_progress_time = (
                        current_time
                        + progress_interval_seconds
                    )

        if progress_callback is not None:
            progress_callback(
                replication,
                total_decisions,
                total_decisions,
                self.config.num_replications,
            )

        return records

    def run(
        self,
        master_seed: int,
        progress_callback: ProgressCallback | None = None,
        progress_interval_seconds: float = 5.0,
    ) -> list[DecisionRecord]:
        """Run all configured replications from one master seed."""
        if master_seed < 0:
            raise ValueError(
                "master_seed must be non-negative."
            )

        if progress_interval_seconds <= 0.0:
            raise ValueError(
                "progress_interval_seconds must be positive."
            )

        records: list[DecisionRecord] = []

        for replication in range(
            self.config.num_replications
        ):
            records.extend(
                self.run_replication(
                    replication=replication,
                    master_seed=master_seed,
                    progress_callback=progress_callback,
                    progress_interval_seconds=(
                        progress_interval_seconds
                    ),
                )
            )

        return records