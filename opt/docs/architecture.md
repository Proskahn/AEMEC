# Architecture

The optimization workflow has narrow dependency boundaries and an intentionally
small public surface.

| Module | Responsibility | May depend on |
| --- | --- | --- |
| `aemec_opt/engine.py` | Resumable multi-objective study, completed-trial budget, trial failures, and generic Pareto selection | Python standard library, Optuna |
| `aemec_opt/reporting.py` | CSV and two-objective Pareto plots | `engine.py`, Matplotlib |
| `aemec_opt/case.py` | AEMEC geometry edits, OpenFOAM execution, log parsing, and objective validation | `engine.py`, Python standard library |
| `aemec_opt/cli.py` | AEMEC command-line configuration and composition | all package modules |
| `run_optimization.py` | Stable executable facade | `aemec_opt.cli` |
| `scripts/` | Post-processing of completed study artifacts | package API or standalone plotting dependencies |

`engine.py` contains no OpenFOAM terminology, case layout, or plotting code. A
different simulator can supply an evaluator that maps a candidate value to an
`ObjectiveResult`, then reuse the engine and optional reporting layer.

## Optimizer engine

The engine is a simulator-independent, one-parameter, multi-objective layer. It
stores Optuna studies in SQLite and minimizes an arbitrary tuple of objective
values. The AEMEC CLI configures membrane thickness as its parameter and cell
voltage plus hydrogen crossover as its two objectives.

The first `--startup-trials` proposals are evenly spaced across the bounds.
Later proposals use Optuna's multi-objective TPE sampler. The requested
`--iterations` value counts completed evaluations; failed simulator calls are
recorded but do not consume the budget. A settings fingerprint prevents
incompatible studies from silently sharing a database.

The reporting layer writes all completed records to
`optimization_results.csv`, non-dominated records to `pareto_front.csv`, and a
two-objective plot to `pareto_front.png`.

## AEMEC adapter

The adapter converts a membrane thickness into a self-contained OpenFOAM
evaluation. It copies `run/AEMEC`, modifies the block-mesh geometry, runs the
mesh and solver commands, and parses the resulting log. The source case is
never modified.

The baseline block mesh has a centred 30 um membrane. For a proposed thickness,
the adapter translates both half-stacks by half of the thickness difference;
the catalyst, porous, channel, and interconnect thicknesses remain unchanged.

Every trial runs the complete 1.3--2.3 V potentiostatic table. The adapter takes
the settled final window from each hold and linearly interpolates the cell
voltage and anode hydrogen-crossover rate at 1 A/cm2. It rejects incomplete,
unstable, multiply crossing, or unbracketed curves instead of extrapolating.

Each completed solver call retains its raw log and writes a compact
polarization-curve CSV and PNG. This diagnostic output is written before final
objective acceptance so rejected curves remain inspectable.

The current adapter supports only the block-mesh workflow. The SALOME geometry
contains separate nominal dimensions and hard-coded identifiers and is not
parameterized.
