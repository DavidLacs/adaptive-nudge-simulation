"""Medium-scale validation run for the adaptive nudge simulation."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from adaptive_nudge.config import SimulationConfig
from adaptive_nudge.experiments import (
    primary_conditions,
    non_stationary_conditions,
)
from adaptive_nudge.runner import run_experiment_suite


def main() -> None:
    """Run a medium-scale validation of the primary and non-stationary conditions."""

    validation_config = replace(
        SimulationConfig(),
        decisions_per_replication=2_000,
        num_replications=10,
    )

    conditions = (
        primary_conditions()
        + non_stationary_conditions()
    )

    print("=" * 72)
    print("ADAPTIVE NUDGE SIMULATION — VALIDATION RUN")
    print("=" * 72)
    print()
    print(
        f"Decisions per replication : "
        f"{validation_config.decisions_per_replication}"
    )
    print(
        f"Replications              : "
        f"{validation_config.num_replications}"
    )
    print(
        f"Conditions                : "
        f"{len(conditions)}"
    )
    print()

    print("Conditions:")
    for condition in conditions:
        print(
            f"  - {condition.name} "
            f"({condition.algorithm}, A{condition.action_space_size})"
        )

    print()
    print("Running validation experiment...")
    print()

    results = run_experiment_suite(
        config=validation_config,
        conditions=conditions,
    )

    print("Validation run completed successfully.")
    print()

    rows: list[dict[str, object]] = []

    for result in results:
        print("-" * 72)
        print(f"CONDITION: {result.condition.name}")
        print("-" * 72)

        for metric_name, metric_value in (
            result.aggregated_metrics.items()
        ):
            print(
                f"{metric_name}: {metric_value}"
            )

        print()

        row: dict[str, object] = {
            "condition": result.condition.name,
            "algorithm": result.condition.algorithm,
            "experiment_type": result.condition.experiment_type,
            "action_space_size": result.condition.action_space_size,
        }

        row.update(result.aggregated_metrics)
        rows.append(row)

    output = pd.DataFrame(rows)

    output_path = "validation_results.csv"

    output.to_csv(
        output_path,
        index=False,
    )

    print("=" * 72)
    print("VALIDATION RUN COMPLETE")
    print("=" * 72)
    print()
    print(f"Saved aggregate results to: {output_path}")
    print()
    print("Result table:")
    print(output.to_string(index=False))
    print()


if __name__ == "__main__":
    main()