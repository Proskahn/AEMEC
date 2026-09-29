# Architecture

The optimization workflow has narrow dependency boundaries and an intentionally
small public surface.

| Module | Responsibility | May depend on |
| --- | --- | --- |
| `aemec_opt/engine.py` | Resumable multi-objective study, completed-trial budget, trial failures, and generic Pareto selection | Python standard library, Optuna |
| `aemec_opt/knee.py` | Normalized Chebyshev and bend-angle knee selection | `engine.py`, Python standard library |
| `aemec_opt/reporting.py` | CSV and annotated two-objective Pareto plots | `engine.py`, `knee.py`, Matplotlib |
| `aemec_opt/case.py` | AEMEC geometry edits, OpenFOAM execution, log parsing, and objective validation | `engine.py`, Python standard library |
| `aemec_opt/cli.py` | AEMEC command-line configuration and composition | all package modules |
| `run_optimization.py` | Stable executable facade | `aemec_opt.cli` |
| `scripts/` | Post-processing of completed study artifacts | package API or standalone plotting dependencies |

`engine.py` contains no OpenFOAM terminology, case layout, or plotting code. A
different simulator can supply an evaluator that maps a parameter dictionary to an
`ObjectiveResult`, then reuse the engine and optional reporting layer.

## Optimizer engine

The engine is a simulator-independent, multi-parameter, multi-objective layer. It
stores Optuna studies in SQLite and minimizes an arbitrary tuple of objective
values. `SearchConfig.parameters` contains named `SearchParameter` bounds;
evaluators receive `(trial_number, {parameter_name: value, ...})`. The AEMEC
CLI configures membrane thickness and common water inlet temperature as its
parameters, and cell voltage plus hydrogen crossover as its two objectives.

The first `--startup-trials` proposals use evenly spaced levels in each
dimension, independently shuffled using the seed. This covers each parameter's
range without coupling thickness and temperature along the search-box diagonal.
Later proposals use Optuna's multi-objective TPE sampler. The requested
`--iterations` value counts completed evaluations; failed simulator calls are
recorded but do not consume the budget. A settings fingerprint prevents
incompatible studies from silently sharing a database.

The reporting layer writes all completed records to
`optimization_results.csv`, non-dominated records to `pareto_front.csv`, and a
two-objective plot to `pareto_front.png`. Objective values are normalized over
the Pareto front with zero at the ideal point. `knee_points.csv` records the
point minimizing normalized Chebyshev distance and, when one exceeds the
configured positive-angle threshold, the maximum bend-angle point. The bend
angle uses the two normalized extreme trade-off points as its left and right
references.
Both design parameters are retained in all three CSVs and in the SQLite
database. The Pareto figure shows the same objective front in two panels,
coloured by thickness and temperature respectively. Legacy thickness-only
CSVs remain usable with the knee and polarization comparison commands.

## AEMEC adapter

The adapter converts membrane thickness and water inlet temperature into an
OpenFOAM evaluation. It copies `run/AEMEC`, modifies the geometry and inlets, runs the
mesh and solver commands, and parses the resulting log. The source case is
never modified.

The baseline block mesh has a centred 30 um membrane. For a proposed thickness,
the adapter translates both half-stacks by half of the thickness difference;
the catalyst, porous, channel, and interconnect thicknesses remain unchanged.

The inlet temperature is written as an explicit `uniform` value on
`anodeInlet` and `cathodeInlet` in `0.orig/T`, and on each region's inlet in
`0.orig/{anode,cathode}/T.{water,gas}`. These fields must agree because the
parent mesh solves the conjugate energy equation and the phases use local
thermal equilibrium. Updating only the water fields would leave the parent
energy inlet at its baseline temperature. Templates are updated before
meshing; generated `0` fields are updated again before the solver starts.
The adapter rejects an isothermal (`solveEnergy false`) source case.

The source-case initial temperature, collector thermal boundaries, outlet
backflow temperatures, and Arrhenius reference coefficients are preserved.
Thus the design variable controls feed temperature, not an imposed uniform
cell temperature. Temperature-dependent conductivity, kinetics, and crossover
use the resulting local temperature fields. The objective contract still
checks settled current/crossover windows; it does not establish that the
entire temperature field has reached steady state. Validate sweep duration
for the chosen thermal boundary conditions when using the results.

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
