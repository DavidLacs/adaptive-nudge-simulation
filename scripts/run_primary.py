"""Run the full primary experiment suite."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from adaptive_nudge.config import SimulationConfig
from adaptive_nudge.experiments import primary_conditions
from adaptive_nudge.runner import run_experiment_suite


def main() -> None:
    """Run the full primary experiment suite and save aggregate results."""

    config = replace(
        SimulationConfig(),
        decisions_per_replication=10_000,
        num_replications=100,
    )

    conditions = primary_conditions()

    print("=" * 72)
    print("ADAPTIVE NUDGE SIMULATION — FULL PRIMARY EXPERIMENT")
    print("=" * 72)
    print()
    print(
        f"Decisions per replication : "
        f"{config.decisions_per_replication}"
    )
    print(
        f"Replications              : "
        f"{config.num_replications}"
    )
    print(
        f"Conditions                : "
        f"{len(conditions)}"
    )
    print(
        f"Total decisions           : "
        f"{config.decisions_per_replication * config.num_replications * len(conditions):,}"
    )
    print()

    print("Conditions:")
    for condition in conditions:
        print(
            f"  - {condition.name} "
            f"({condition.algorithm}, A{condition.action_space_size})"
        )

    print()
    print("Running full primary experiment...")
    print()

    results = run_experiment_suite(
        config=config,
        conditions=conditions,
    )

    rows: list[dict[str, object]] = []

    for result in results:
        row: dict[str, object] = {
            "condition": result.condition.name,
            "experiment_type": result.condition.experiment_type,
            "algorithm": result.condition.algorithm,
            "action_space_size": result.condition.action_space_size,
        }

        for metric_name, metric_value in result.aggregated_metrics.items():
            row[metric_name] = metric_value

        rows.append(row)

    output = pd.DataFrame(rows)

    output_path = "primary_results.csv"
    output.to_csv(
        output_path,
        index=False,
    )

    print("=" * 72)
    print("FULL PRIMARY EXPERIMENT COMPLETE")
    print("=" * 72)
    print()
    print(f"Saved aggregate results to: {output_path}")
    print()
    print("Result table:")
    print(output.to_string(index=False))
    print()


if __name__ == "__main__":
    main()