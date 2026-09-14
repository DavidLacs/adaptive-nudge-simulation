# Adaptive Nudge Simulation

Simulation-based evaluation of a contextual bandit approach to adaptive
nudge timing in an infinite-scroll social-media-inspired environment.

## Research Methods

The simulation evaluates:

- **LinUCB** as the primary contextual bandit method;
- **UCB1** as a non-contextual adaptive baseline;
- **Fixed 10-minute timing** as a static baseline.

The simulation uses synthetic session duration, contextual state,
intervention timing, and intervention-response generation according to the
methodology specified in the thesis.

The simulation is evaluated across independent replications using
replication-level performance metrics and 95% confidence intervals.

## Project Structure

```text
src/adaptive_nudge/
    __init__.py
    algorithms.py
    config.py
    environment.py
    evaluation.py
    results.py
    simulation.py

tests/
    test_environment.py
    test_results.py