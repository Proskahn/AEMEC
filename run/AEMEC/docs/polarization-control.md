# Polarization-curve control

## What was wrong with the old scan

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

## Corrected sequence

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

The stable controller is configured in
`constant/phiEAnode/regionProperties` under `galvanostatic.polarizationCurve`:

| Parameter | Meaning | AEMEC starting value |
| --- | --- | --- |
| `targets` | Signed collector current-density targets, A/m2 | `(-6000 -8000 -10000 -12500 -15000)` |
| `minimumHoldDuration` | Minimum time before a point can be accepted, s | `30` |
| `targetCurrentTolerance` | Relative current-target error | `0.05` |
| `currentScale` | A/m2 scale used near zero target | `100` |
| `voltageTolerance` | Consecutive-sample voltage tolerance, V | `0.002` |
| `currentStabilityTolerance` | Consecutive-sample current variation relative to scale | `0.02` |
| `stabilitySamples` | Consecutive stable samples required | `5` |

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

After rebuilding the solver, run the case long enough for the stable holds:

```bash
cd /root/work/AEMEC
./src/Allwmake
cd run/AEMEC
make clear
make mesh
make srun
```

The supplied `controlDict.run` allows 600 s. Adjust `endTime` upward if an
accepted point has not yet been reached; the controller will not advance an
unsettled target.

The extractor now plots only controller records marked `accepted: true`:

```bash
python3 visualization/polarized_curve.py --log run/AEMEC/log.run
```

Use `--all-samples` only to inspect controller transients. For old logs that
do not contain the new controller record, the script retains only samples that
meet the same 5% target-current check by default.
