"""Small end-to-end smoke run for the adaptive nudge simulation."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from adaptive_nudge.config import SimulationConfig
from adaptive_nudge.experiments import primary_conditions
from adaptive_nudge.runner import run_experiment_suite


def main() -> None:
    """Run a small version of the primary experiment suite."""

    # ------------------------------------------------------------------
    # Smoke-test configuration
    #
    # These values are intentionally much smaller than the thesis
    # configuration. The purpose is to verify the complete execution
    # and evaluation pipeline before launching the full experiment.
    # ------------------------------------------------------------------
    smoke_config = replace(
        SimulationConfig(),
        decisions_per_replication=200,
        num_replications=3,
    )

    conditions = primary_conditions()

    print("=" * 72)
    print("ADAPTIVE NUDGE SIMULATION — END-TO-END SMOKE RUN")
    print("=" * 72)
    print()
    print(
        f"Decisions per replication : "
        f"{smoke_config.decisions_per_replication}"
    )
    print(
        f"Replications              : "
        f"{smoke_config.num_replications}"
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
    print("Running...")
    print()

    results = run_experiment_suite(
        config=smoke_config,
        conditions=conditions,
    )

    print("Run completed successfully.")
    print()

    # ------------------------------------------------------------------
    # Display the aggregated metrics exactly as returned by the
    # results/evaluation pipeline.
    # ------------------------------------------------------------------
    rows: list[dict[str, object]] = []

    for result in results:
        print("-" * 72)
        print(f"CONDITION: {result.condition.name}")
        print("-" * 72)

        for metric_name, metric_value in result.aggregated_metrics.items():
            print(
                f"{metric_name}: {metric_value}"
            )

        print()

        row: dict[str, object] = {
            "condition": result.condition.name,
            "algorithm": result.condition.algorithm,
            "action_space_size": result.condition.action_space_size,
        }

        row.update(result.aggregated_metrics)
        rows.append(row)

    # ------------------------------------------------------------------
    # Create a small CSV containing the smoke-run aggregate results.
    #
    # The CSV is only a diagnostic artifact at this stage. It is not
    # the final thesis result dataset.
    # ------------------------------------------------------------------
    output = pd.DataFrame(rows)

    output_path = "smoke_results.csv"
    output.to_csv(
        output_path,
        index=False,
    )

    print("=" * 72)
    print("SMOKE RUN COMPLETE")
    print("=" * 72)
    print()
    print(f"Saved aggregate results to: {output_path}")
    print()
    print("Result table:")
    print(output.to_string(index=False))
    print()


if __name__ == "__main__":
    main()