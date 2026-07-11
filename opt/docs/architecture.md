# Architecture

The optimization workflow has intentionally narrow dependency boundaries.

| Module | Responsibility | May depend on |
| --- | --- | --- |
| `optimization_lib.py` | Resumable multi-objective study, completed-trial budget, trial failures, generic Pareto mask | Python standard library, Optuna |
| `reporting.py` | CSV and two-objective Pareto plot generation | `optimization_lib.py`, Matplotlib |
| `aemec_case.py` | AEMEC geometry edits, OpenFOAM controls, subprocess execution, log parsing, objective validation | `optimization_lib.py`, Python standard library |
| `cli.py` | AEMEC command-line configuration and composition of the three layers | all modules above |
| `run_optimization.py` | Backward-compatible executable facade | `cli.py`, `optimization_lib.py` |

`optimization_lib.py` does not contain OpenFOAM terms, filesystem case layout,
or plotting code. A different simulator should supply an evaluator that maps a
candidate value to `ObjectiveResult`, then reuse the library and optionally its
reporting module.

The AEMEC-specific adapter receives the chosen thickness in micrometres and
returns two minimization objectives: cell voltage and hydrogen crossover rate.
It owns all operating-point and convergence checks, so the optimizer never
needs to interpret solver logs.
