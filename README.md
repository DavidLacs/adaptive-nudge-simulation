# Adaptive Nudge Simulation

Simulation-based evaluation of a contextual bandit approach to adaptive
nudge timing in an infinite-scroll social-media-inspired environment.

## Research methods

The simulation evaluates:

- LinUCB as the primary contextual bandit method;
- UCB1 as a non-contextual adaptive baseline;
- a fixed 10-minute timing baseline.

The simulation uses synthetic session duration, contextual state,
intervention timing, and intervention-response generation according to the
methodology specified in the thesis.

## Project structure

```text
src/adaptive_nudge/
    config.py
    environment.py
    algorithms.py
    oracle.py
    simulation.py
    metrics.py
    results.py