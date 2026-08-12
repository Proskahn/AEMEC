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

1. Cell voltage at a current magnitude of 1 A/cm2 (`10000 A/m2`).
2. Anode-side hydrogen crossover/release rate at that same interpolated
   operating point, in `mol/s`.

Each trial retains potentiostatic control and runs the complete 1.3--2.3 V
table in 0.1 V increments. The adapter takes the settled final window from
each voltage hold, finds the two adjacent current-density samples surrounding
1 A/cm2, and applies linear interpolation. It rejects the trial rather than
extrapolating when the simulated curve does not bracket 1 A/cm2.

The crossover objective is the positive part of the signed anode catalyst-layer
desorption integral,
`integral[kLa_anode*(cH2 - H_anode*p*XH2) dV]`. The connected dissolved field
spans both catalyst layers and the membrane. With the default
`faradaicDissolvedFraction = 1`, Faradaic H2 first enters that field and the
same local amount is subtracted from the direct cathode-gas reaction source.
Both CL gas-transfer terms are applied locally with equal and opposite terms
in the dissolved equation. The printed three-continuum coupling imbalance
must remain near round-off.

Every solver completion also writes
`logs/trial_XXXX_polarization_curve.csv` and a matching
`logs/trial_XXXX_polarization_curve.png`. The CSV contains one row per voltage
hold with the final current, current density, crossover rate, final-window
mean/CV/drift statistics, and interpolation-endpoint weight. It is written
before objective acceptance, so a completed simulation remains inspectable
even if its curve is later rejected for stability or bracketing.

## Sweep duration

The supplied voltage table has eleven 15 s holds, so each CFD evaluation runs
to 165 s at `deltaT = 0.1 s`. The legacy fast/ramp and solver-iteration command
line options remain accepted for command compatibility, but they do not
shorten or replace this potentiostatic sweep.

## Acceptance checks

The adapter uses post-solve collector current-density records, not a pre-solve
reaction-source value. It requires a complete final window at every voltage
hold, then applies the stability criteria only to the exact target point or the
two holds used for interpolation. This avoids rejecting a useful curve because
an unused low-voltage crossover value is close to zero. A trial is rejected for
an incomplete curve, an unbracketed target, multiple crossings, unstable
interpolation endpoints, command failure, or abnormal solver termination.
Defaults require five final samples, <=5% current coefficient of variation,
<=0.005 V voltage variation, and <=2% crossover variation at the objective
endpoints. The current coefficient of variation is the population standard
deviation divided by the final-window mean; unlike a peak-to-peak range, it
does not reject a bounded five-sample oscillation merely because both extrema
occur in the window.

Only the block-mesh workflow is parameterized. `make salomeMesh` is unsupported
because its SALOME geometry has separate nominal thickness and hard-coded IDs.
