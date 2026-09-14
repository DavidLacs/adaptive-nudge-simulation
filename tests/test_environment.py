"""Tests for the synthetic simulation environment."""

from __future__ import annotations

import math

import numpy as np
import pytest

from adaptive_nudge.config import (
    ACTION_SPACES,
    GROUND_TRUTH_COEFFICIENTS,
    SimulationConfig,
)
from adaptive_nudge.environment import Environment


def make_environment(
    config: SimulationConfig | None = None,
) -> Environment:
    """Create an environment with deterministic test RNG streams."""
    if config is None:
        config = SimulationConfig()

    return Environment(
        config=config,
        duration_rng=np.random.default_rng(0),
        context_rng=np.random.default_rng(1),
    )


def test_default_scaling_preserves_original_pre_change_coefficients() -> None:
    """The default scaling of 1.0 must preserve the thesis coefficients."""
    environment = make_environment()

    coefficients = environment.coefficients_for_decision(
        decision_index=0
    )

    assert coefficients == GROUND_TRUTH_COEFFICIENTS


def test_temporal_scaling_08_scales_only_temporal_coefficients() -> None:
    """Scaling 0.8 must affect only the four temporal coefficients."""
    config = SimulationConfig(
        temporal_coefficient_scaling=0.8
    )
    environment = make_environment(config)

    coefficients = environment.coefficients_for_decision(
        decision_index=0
    )

    for timing, original in GROUND_TRUTH_COEFFICIENTS.items():
        expected = (
            original[0] * 0.8,
            original[1] * 0.8,
            original[2] * 0.8,
            original[3] * 0.8,
            original[4],
        )

        assert coefficients[timing] == pytest.approx(
            expected
        )


def test_temporal_scaling_12_scales_only_temporal_coefficients() -> None:
    """Scaling 1.2 must affect only the four temporal coefficients."""
    config = SimulationConfig(
        temporal_coefficient_scaling=1.2
    )
    environment = make_environment(config)

    coefficients = environment.coefficients_for_decision(
        decision_index=0
    )

    for timing, original in GROUND_TRUTH_COEFFICIENTS.items():
        expected = (
            original[0] * 1.2,
            original[1] * 1.2,
            original[2] * 1.2,
            original[3] * 1.2,
            original[4],
        )

        assert coefficients[timing] == pytest.approx(
            expected
        )

        assert coefficients[timing][4] == pytest.approx(
            0.03
        )


def test_post_change_reverses_scaled_temporal_coefficients() -> None:
    """The post-change regime must reverse the scaled temporal coefficients."""
    config = SimulationConfig(
        temporal_coefficient_scaling=1.2
    )
    environment = make_environment(config)

    post_change = environment.coefficients_for_decision(
        decision_index=config.regime_change_point
    )

    for timing, original in GROUND_TRUTH_COEFFICIENTS.items():
        expected = (
            original[3] * 1.2,
            original[2] * 1.2,
            original[1] * 1.2,
            original[0] * 1.2,
            original[4],
        )

        assert post_change[timing] == pytest.approx(
            expected
        )

        assert post_change[timing][4] == pytest.approx(
            0.03
        )


def test_regime_change_uses_pre_change_before_boundary_and_post_change_at_boundary() -> None:
    """The coefficient regime must change exactly at the configured boundary."""
    config = SimulationConfig(
        decisions_per_replication=100,
        regime_change_fraction=0.5,
        temporal_coefficient_scaling=1.2,
    )
    environment = make_environment(config)

    pre_change = environment.coefficients_for_decision(
        decision_index=49
    )
    post_change = environment.coefficients_for_decision(
        decision_index=50
    )

    assert pre_change[20] == pytest.approx(
        (
            0.08 * 1.2,
            0.25 * 1.2,
            0.35 * 1.2,
            0.72 * 1.2,
            0.03,
        )
    )

    assert post_change[20] == pytest.approx(
        (
            0.72 * 1.2,
            0.35 * 1.2,
            0.25 * 1.2,
            0.08 * 1.2,
            0.03,
        )
    )


def test_specified_temporal_scaling_values_produce_valid_response_probabilities() -> None:
    """All specified scaling conditions must keep response probabilities valid."""
    context_by_state = {
        "morning": np.array(
            [1.0, 0.0, 0.0, 0.0, 1.0]
        ),
        "afternoon": np.array(
            [0.0, 1.0, 0.0, 0.0, 1.0]
        ),
        "evening": np.array(
            [0.0, 0.0, 1.0, 0.0, 1.0]
        ),
        "late_night": np.array(
            [0.0, 0.0, 0.0, 1.0, 1.0]
        ),
    }

    for scaling in (0.8, 1.0, 1.2):
        config = SimulationConfig(
            temporal_coefficient_scaling=scaling
        )
        environment = make_environment(config)

        for decision_index in (
            0,
            config.regime_change_point,
        ):
            for timing in ACTION_SPACES[8]:
                for context in context_by_state.values():
                    probability = environment.response_probability(
                        timing_minutes=timing,
                        context=context,
                        decision_index=decision_index,
                    )

                    assert math.isfinite(probability)
                    assert 0.0 <= probability <= 1.0


def test_invalid_temporal_coefficient_scaling_is_rejected() -> None:
    """Non-positive or non-finite scaling values must be rejected."""
    for scaling in (0.0, -0.1, math.nan, math.inf, -math.inf):
        config = SimulationConfig(
            temporal_coefficient_scaling=scaling
        )

        with pytest.raises(ValueError):
            make_environment(config)