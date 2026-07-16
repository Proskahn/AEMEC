# Polarization-curve control

## Default: convergence-based current sweep

The default case uses galvanostatic control (`galvanostatic.active true`) and
enables `polarizationCurve`. It requests current-density magnitudes from `0.0`
to `2.0 A/cm2` in `0.2 A/cm2` increments. Electrolysis current is negative in
the solver convention, so the configured targets are
`0, -2000, ..., -20000 A/m2`. The collector voltage is adjusted until each
current target is accepted.

The controller requires both the current-target error and consecutive current
and voltage changes to satisfy the configured tolerances. It advances only
after the minimum 15-second hold and five stable samples. The `ibar` time table
is retained only as a fallback for runs that disable `polarizationCurve`.

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

## Convergence-based galvanostatic sequence

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
| `targets` | Signed collector current-density targets, A/m2 | `(0 -2000 ... -20000)` |
| `minimumHoldDuration` | Minimum time before a point can be accepted, s | `15` |
| `targetCurrentTolerance` | Relative current-target error | `0.05` |
| `currentScale` | A/m2 scale used near zero target | `100` |
| `voltageTolerance` | Consecutive-sample voltage tolerance, V | `0.002` |
| `currentStabilityTolerance` | Consecutive-sample current variation relative to scale | `0.02` |
| `stabilitySamples` | Consecutive stable samples required | `5` |

For this scan, `maxVoltageStep` is `0.01 V` per outer
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

Run the default current sweep:

```bash
cd /Users/zhuang/AEMEC
cd run/AEMEC
make clear
make mesh
make srun
```

The supplied `controlDict.run` has a 300-second safety timeout. The solver
stops earlier and writes the final multi-region state after all 11 targets are
accepted. If the run reaches 300 seconds first, inspect the last target for a
voltage-bound or convergence failure before increasing the timeout.

From the repository root, extract the controller records marked as accepted:

```bash
python3 visualization/polarized_curve.py \
    --log run/AEMEC/log.run \
    --scan-mode current
```

The extractor plots only controller records marked `accepted: true`; use
`--all-samples` only to inspect rejected transients. For a fixed-voltage run,
use `prepare_fixed_voltage_diagnostic.py`, which disables galvanostatic
feedback in an isolated case copy.
