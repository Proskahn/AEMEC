# Fixed-current CFD evaluation

The default evaluator holds **1 A/cm²** for **20 solver-clock units**, then
returns the mean cell voltage [V] and hydrogen crossover rate [mol/s] over
the final **5 clock units**. These are direct CFD observations at the operating
current. The previous mode interpolates these objectives from a complete
1.3–2.3 V sweep and remains available with `--evaluation-mode voltage-sweep`.

The same evaluator is used by `opt`, `benchmark`, and `ELA evaluate`. This
change does not alter the three design variables, objective units, or the
requirement to evaluate every benchmark proposal with CFD.

## Commands

Run all four baseline algorithms with the new default:

```bash
python benchmark/run_benchmark.py \
  --algorithms gp tpe nsga2 cmaes \
  --budget 64 --initial 16 --seeds 42 43 44 45 46 \
  --evaluation-mode fixed-current \
  --simulation-duration-s 20 --final-window-s 5 \
  --output benchmark/results/fixed_current_20
```

For a 10-unit run, change `--simulation-duration-s 20` to
`--simulation-duration-s 10` and select a fresh output directory. Add
`--dry-run` to preview the protocol and evaluation count without calling CFD.
Every algorithm in one benchmark uses the same duration and window.

The same options work for a resumable optimization study:

```bash
python opt/run_optimization.py --iterations 50 \
  --simulation-duration-s 20 --final-window-s 5 \
  --study-name fixed-current-20
```

Or evaluate a fixed sampling design for ELA:

```bash
python ELA/ela.py evaluate ELA/results/design.csv \
  --simulation-duration-s 20 --final-window-s 5 \
  --output-dir ELA/results/fixed_current_20
```

## What changes in the scratch case

The adapter enables galvanostatic control, sets its single signed current
target to **−10,000 A/m²** (the electrolysis convention), and sets both the
controller's minimum hold and `endTime` to the requested duration. `startTime`
is zero. It preserves the source controller gain, voltage limits, initial
potential, and `controlDict.run` time step. The active controls are reapplied
after meshing, which may recreate `controlDict`.

The source case remains unchanged. With the current `deltaT = 0.1`, a 20-unit
run has 200 outer iterations and a 10-unit run has 100, compared with 1,650
for the old sweep. These ratios describe iteration counts, not guaranteed
wall-clock speedups: meshing, per-step convergence and reporting also cost time.

The case preflight now validates the configured current target, hold length,
tolerances and table endpoint instead of requiring the original literal
20-unit settings. It also accepts the source case's existing constant membrane
conductivity (8 S/m), while retaining validation for the legacy Arrhenius
option. No conductivity coefficients or physical dictionaries were changed.

The case uses localEuler/steadyState schemes. The CLI suffix `-s` follows the
case's existing time notation; it does not imply a resolved physical transient.
`duration_s` in result metadata continues to mean measured wall-clock duration.
`simulation_duration_s` records the configured solver-clock endpoint.

## Acceptance and averaging

Both duration and averaging-window length must be positive multiples of the
source time step. The window must be shorter than the run and contain at least
two samples and at least `--stability-samples` samples. For the defaults, the
inclusive interval [15, 20] contains 51 paired records at increments of 0.1.

The evaluator requires normal solver termination and a complete, consecutive
final window. Over that entire window it checks:

| Quantity | Default limit | CLI option |
| --- | --- | --- |
| Signed current error relative to −10,000 A/m², at every sample | 1% | `--target-current-relative-tolerance` |
| Current range / target magnitude | 5% | `--current-relative-tolerance` |
| Voltage maximum minus minimum | 0.005 V | `--voltage-stability-tolerance-v` |
| Crossover range / maximum crossover | 2% | `--crossover-stability-relative-tolerance` |

The target-current test also rejects a stable wrong current or reversed sign.
Voltages must be positive and crossover must be nonnegative; an identically
zero crossover window is valid. Range checks detect drift accumulated over
the window. Passing only the controller's adjacent-sample current check is
insufficient. The independent window checks are authoritative, including when
an older controller reports acceptance one step late.

For accepted trials, arithmetic means of voltage and crossover are the two
objectives. Metadata includes the measured current, window endpoints/count,
target error, ranges and standard deviations. Failed trials do not return
objectives or trigger a hidden longer run; benchmark failures still consume
one evaluation attempt.

Each completed solver call writes `trial_XXXX_timeseries.csv` and a three-panel
`trial_XXXX_timeseries.png` showing current, voltage and crossover, with the
averaging window shaded. These diagnostics are retained when endpoint checks
fail. Fixed-current runs do not produce a polarization curve. Explicit sweep
runs retain their original per-hold CSV/PNG and interpolation checks.

## Compare durations before production use

A stable current after five clock units is evidence for trying a shorter
run, but does not establish that crossover, voltage or temperature fields
have converged throughout the design space. Compare the same designs at
10 and 20 units, including bounds/corners in thickness, temperature and
porosity. Inspect both objective drift and differences between final-window
means. A few millivolts in voltage and about 2% in crossover are possible
starting acceptance criteria; choose them for the intended study accuracy.
Longer runs may be needed for some designs. The code tests use mocked solver
output and cannot validate CFD convergence.

Use fresh output directories/study names when changing modes, duration or
window. These settings and the objective schema version are persisted, so
resuming a pre-change sweep study or a different fixed-current protocol fails
instead of combining incompatible observations. No existing results are
rewritten. Run the sweep explicitly on matching designs when checking the
effect of replacing interpolation with direct fixed-current measurements.
