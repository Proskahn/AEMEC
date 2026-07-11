# Optimizer library

`optimization_lib.py` is a simulator-independent, one-parameter,
multi-objective optimization layer. It stores Optuna studies in SQLite and
minimizes an arbitrary tuple of objective values.

The AEMEC CLI configures it with membrane thickness as the parameter and two
objectives, but the library itself has no AEMEC or OpenFOAM dependency.

## Search and persistence

The first `--startup-trials` proposals are evenly spaced across the configured
bounds. Later proposals use Optuna's multi-objective TPE sampler. The default
seed is 42. A study directory is derived from the study name and the complete
settings fingerprint, preventing incompatible configurations from silently
sharing a database.

The requested `--iterations` count is a count of **completed** evaluations.
Failed simulator calls are recorded as failed trials and do not consume that
budget. The run aborts after `--max-consecutive-failures` failures, then may be
resumed after the environment is fixed.

Rerun exactly the same command to resume. Bounds, seed, study settings, source
case identity, and adapter validation settings are checked before continuation.
Use a new `--study-name` when deliberately changing a configuration.

## Results

The reporting layer writes all completed records to
`optimization_results.csv`, non-dominated records to `pareto_front.csv`, and a
two-objective scatter plot to `pareto_front.png`. Duplicate objective points are
preserved as Pareto points; a point is removed only when another point is no
worse in every objective and strictly better in at least one.

To use the library with another simulator, write a small evaluator that
accepts `(trial_number, parameter_value)` and returns an `ObjectiveResult`.
Keep simulator setup, execution, parsing, and physics validation in that
adapter rather than adding them to the library.
