# Polarization-curve control

## Default on the crossover branch: fixed current

The default case uses galvanostatic control (`galvanostatic.active true`) at
`-10000 A/m2`, corresponding to an electrolysis-current magnitude of
`1 A/cm2`. The applied collector voltage starts from `2.0 V`, is corrected from
the post-solve collector-current flux, and is held for a 20 s crossover
experiment. The single target is accepted only after the 20 s minimum hold and
a stable five-sample final window.

The retained `1.3--2.3 V` table is not active in the base crossover case. The
optimization adapter explicitly disables galvanostatic feedback in its scratch
copy and restores the complete 165 s voltage sweep used to interpolate its
objectives at `1 A/cm2`.

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

## Galvanostatic controller

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

The stable current controller is configured in
`constant/phiEAnode/regionProperties` under `galvanostatic.polarizationCurve`:

| Parameter | Meaning | AEMEC starting value |
| --- | --- | --- |
| `targets` | Signed collector current-density targets, A/m2 | `(-10000)` |
| `minimumHoldDuration` | Minimum time before the point can be accepted, s | `20` |
| `targetCurrentTolerance` | Relative current-target error | `0.01` |
| `currentScale` | A/m2 scale used near zero target | `100` |
| `voltageTolerance` | Consecutive-sample voltage tolerance, V | `0.002` |
| `currentStabilityTolerance` | Consecutive-sample current variation relative to scale | `0.01` |
| `stabilitySamples` | Consecutive stable samples required | `5` |

For this experiment, the proportional controller gain is `relax = 1e-6` and
`maxVoltageStep` is `0.002 V` per outer iteration. These values suppress the
two-point current oscillation observed with `relax = 1e-5` and a `0.01 V` step
near the 0.6 A/cm2 target, while retaining the existing 5%-target and stability
checks.

The fixed-current case does not specify `minVoltage` or `maxVoltage`, so the
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

Run the fixed-current crossover case:

```bash
cd /Users/zhuang/AEMEC
cd run/AEMEC
make clear
make mesh
make srun
```

The supplied `controlDict.run` ends at 20 s and writes the final state.

Inspect all current-controller samples with:

```bash
python3 visualization/polarization_curve.py \
    --log run/AEMEC/log.run \
    --scan-mode current \
    --all-samples
```

Without `--all-samples`, the extractor keeps only controller records marked
`accepted: true`. The final target is accepted at 20 s only if its current and
voltage stability requirements are satisfied.
