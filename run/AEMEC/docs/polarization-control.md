# Polarization-curve control

## Default: one-run voltage sweep

The default case uses potentiostatic control (`galvanostatic.active false`) to
avoid feedback tuning. A `voltage` table holds `1.3, 1.5, 1.7, 1.9, 2.1,` and
`2.3 V` for 15 seconds each. The current at the end of each plateau supplies
one polarization point. The `ibar` table and `polarizationCurve.targets` remain
in the dictionary for optional galvanostatic and optimizer runs, but neither is
active during the default voltage sweep.

This mode is intentionally time-based. Before accepting the resulting curve,
check that the collector current and residuals have settled near the end of
each hold. Increase the hold duration if they have not.

## Why the former current scan did not produce a curve

The former controller updated its voltage in `electric::correct()` using the
integral of the volumetric reaction-source field `J`. That happens before the
electric potential solve. The plotted value, however, was the collector flux
computed later in `electric::solve()`. Consequently the feedback current and
the logged collector current were different quantities from different stages
of the outer iteration.

The former AEM schedule compounded this with a ten-second time table. It
advanced through targets whether or not the controller had converged. The
`maxVoltageStep = 0.005 V` limit moved the voltage by at most `0.5 V` in a
ten-second hold at `deltaT = 0.1 s`. When the controller requested a voltage
outside `[minVoltage, maxVoltage] = [1.0, 2.5] V`, the boundary value saturated
but the time table continued. The resulting log therefore mixed a requested
target with a transient or saturated boundary current.

`-10000 A/m2` is a requested 1 A/cm2 operating point in the legacy table, not
a controller current limit. The kinetic model separately bounds its local
volumetric source at `jMax = 5e8 A/m3`; it now reports lower- and upper-clipped
cell counts every update.

## Optional convergence-based galvanostatic sequence

Each outer iteration now performs:

```text
retain active target and previously applied collector voltage
→ solve transport/electrochemistry/electric potential
→ measure the post-solve collector current flux
→ assess target error and stability
→ log the paired current/voltage state
→ apply the voltage correction for the next iteration
→ advance to the next target only after acceptance
```

The optional stable current controller is configured in
`constant/phiEAnode/regionProperties` under `galvanostatic.polarizationCurve`:

| Parameter | Meaning | AEMEC starting value |
| --- | --- | --- |
| `targets` | Signed collector current-density targets, A/m2 | `(-6000 -9000 -12000 -15000)` |
| `minimumHoldDuration` | Minimum time before a point can be accepted, s | `15` |
| `targetCurrentTolerance` | Relative current-target error | `0.05` |
| `currentScale` | A/m2 scale used near zero target | `100` |
| `voltageTolerance` | Consecutive-sample voltage tolerance, V | `0.002` |
| `currentStabilityTolerance` | Consecutive-sample current variation relative to scale | `0.02` |
| `stabilitySamples` | Consecutive stable samples required | `5` |

For this proof-of-concept scan, `maxVoltageStep` is `0.01 V` per outer
iteration. This reduces controller ramp time while retaining the existing
5%-target and stability checks.

The controller rejects a point when the voltage is at `minVoltage` or
`maxVoltage`, or when applying the correction would clip the voltage. It keeps
that target active instead of silently advancing. If a target remains rejected
until `endTime`, treat it as unreachable within the selected voltage bounds;
do not put it on the polarization curve.

Every controller update writes one `galvanostatic target:` line with signed and
requested target, measured current, error, applied voltage, raw/limited voltage
step, step/bound clipping flags, planned hold interval, stability count, and
acceptance state. `currentClipped: false` means the controller itself does not
clip current; consult the `ButlerVolmer current limits` line for kinetic-source
clipping.

## Running and plotting

Run the default voltage sweep:

```bash
cd /Users/zhuang/AEMEC
cd run/AEMEC
make clear
make mesh
make srun
```

The supplied `controlDict.run` ends at 90 s, matching the final voltage-table
entry. Increase both the plateau times and `endTime` if 15 s is insufficient
for convergence.

From the repository root, group the logged boundary-current records by voltage:

```bash
python3 visualization/polarized_curve.py \
    --log run/AEMEC/log.run \
    --scan-mode voltage \
    --hold-duration 15
```

For an optional galvanostatic run, set `galvanostatic.active true`, enable
`polarizationCurve`, and choose reachable current targets. In that mode the
extractor plots only controller records marked `accepted: true`; use
`--all-samples` only to inspect rejected transients.
