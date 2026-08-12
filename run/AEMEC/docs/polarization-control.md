# Polarization-curve control

## Default: voltage-table sweep

The default case uses potentiostatic control (`galvanostatic.active false`). A
`voltage` table prescribes `1.3` through `2.3 V` in `0.1 V` increments, holding
each level for 15 s. The resulting collector current density is measured at
the end of each hold. The complete sweep lasts 165 s.

This sweep is time-based rather than convergence-based. Verify that the current
and transport fields have settled by the end of each hold before interpreting
the extracted curve.

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
a controller current limit. The kinetic model retains a numerical safeguard
at `jMax = 2e9 A/m3` (4 A/cm2 for the 20 um catalyst layers); it reports lower-
and upper-clipped cell counts every update. This safeguard is not a physical
limiting-current model.

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
| `targets` | Signed collector current-density targets, A/m2 | `(0 -2000 ... -20000)` |
| `minimumHoldDuration` | Minimum time before a point can be accepted, s | `15` |
| `targetCurrentTolerance` | Relative current-target error | `0.05` |
| `currentScale` | A/m2 scale used near zero target | `100` |
| `voltageTolerance` | Consecutive-sample voltage tolerance, V | `0.002` |
| `currentStabilityTolerance` | Consecutive-sample current variation relative to scale | `0.02` |
| `stabilitySamples` | Consecutive stable samples required | `5` |

For this scan, the proportional controller gain is `relax = 1e-6` and
`maxVoltageStep` is `0.002 V` per outer iteration. These values suppress the
two-point current oscillation observed with `relax = 1e-5` and a `0.01 V` step
near the 0.6 A/cm2 target, while retaining the existing 5%-target and stability
checks.

The optional current scan does not specify `minVoltage` or `maxVoltage`, so the
controller uses its effectively unbounded built-in defaults. The per-update
`maxVoltageStep` remains active. If a target does not meet the current and
stability criteria before `endTime`, it is not accepted and must not be put on
the polarization curve.

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

The supplied `controlDict.run` ends at 165 s, matching the last voltage-table
entry.

From the repository root, extract the final sample from every voltage hold:

```bash
python3 visualization/polarization_curve.py \
    --log run/AEMEC/log.run \
    --scan-mode voltage \
    --hold-duration 15
```

For an optional current-controlled run, enable both `galvanostatic` and
`polarizationCurve`. In that mode the extractor plots only controller records
marked `accepted: true`; use `--all-samples` only to inspect rejected
transients.
