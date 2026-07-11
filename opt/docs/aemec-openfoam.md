# AEMEC OpenFOAM adapter

`aemec_case.py` converts a proposed membrane thickness into a self-contained
OpenFOAM evaluation. It copies `run/AEMEC`, edits only the documented
`blockMeshDict` geometry, runs `make mesh`, runs `openFuelCell`, and parses the
solver log. The source case is not modified.

## Geometry and objectives

The baseline block mesh has a centred 30 um membrane. For a new thickness, the
adapter translates the positive and negative half-stacks by half of the
thickness difference. Catalyst, porous, channel, and interconnect thicknesses
remain unchanged; only membrane and total-stack thickness change.

The two minimization objectives are:

1. Cell voltage at a requested current magnitude of 1 A/cm2 (scheduled as
   `-10000 A/m2` for this case's electrolysis sign convention).
2. Anode-side hydrogen crossover/release rate in `mol/s`.

The crossover objective is the positive anode sink integral
`sinkCoeff * max(cH2 - cH2Anode, 0)`. It represents hydrogen transported
through the membrane and released to the anode; it is not the older Faradaic
hydrogen-generation diagnostic. It requires a positive `sinkCoeff` and a
configured `sinkZone`.

## Fast and ramp modes

`--run-mode fast` is the default. It applies the target from the first outer
iteration, uses 250 outer coupling iterations by default, and writes fields
only at the final iteration. This reduces solver-loop work versus the legacy
roughly 801-iteration ramp, while retaining the air-region local-time-stepping
configuration. `--iteration-clock-step` controls only the current-table and
outer-loop clock; it is not the air-region physical/local pseudo-time step.

Use `--run-mode ramp` when validating the shortcut or when fast mode does not
meet the stability checks. It retains the source current steps before 1 A/cm2,
then holds the target for 30 additional seconds and removes later steps.

## Acceptance checks

The adapter uses the post-solve collector current-density log record, not a
pre-solve boundary value. It selects the final paired voltage and crossover
sample at the target hold and rejects a trial when the current tolerance,
controller motion, voltage stability, crossover stability, command status, or
normal solver termination fails. Defaults require five final samples, <=5%
current error, <=0.005 V voltage range, <=2% crossover range, and <=0.001 V
controller movement.

Only the block-mesh workflow is parameterized. `make salomeMesh` is unsupported
because its SALOME geometry has separate nominal thickness and hard-coded IDs.
