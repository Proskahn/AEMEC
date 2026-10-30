# Runbook

## Setup

Build the solver and install the Python packages:

```bash
cd /path/to/AEMEC/src
./Allwmake

cd /path/to/AEMEC
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r opt/requirements.txt
```

Before running, make sure `openFuelCell`, `blockMesh`, `topoSet`, and the
repository's custom OpenFOAM tools are on `PATH`. Rebuild the solver after the
crossover-objective logging change.

## Run a study

The default 50-evaluation study jointly searches thickness 10–100 µm,
common anode/cathode water inlet temperature 298.15–353.15 K (25–80°C),
and anode PTL porosity 0.40–0.80:

```bash
python3 opt/run_optimization.py \
  --iterations 50 \
  --min-thickness-um 10 \
  --max-thickness-um 100 \
  --min-water-inlet-temperature-k 298.15 \
  --max-water-inlet-temperature-k 353.15 \
  --min-ptl-porosity 0.40 \
  --max-ptl-porosity 0.80 \
  --ptl-side anode
```

The bounds are engineering inputs, not solver-inferred material limits. Use a
smaller pilot before a full study. Temperature arguments are in kelvin and
must be positive, finite, and strictly ordered:

```bash
python3 opt/run_optimization.py \
  --iterations 20 --startup-trials 6 \
  --study-name aemec-fast-pilot
```

Check sample thickness/temperature/porosity combinations with a small study before a long run:

```bash
python3 opt/run_optimization.py --iterations 3 --startup-trials 3 \
  --simulation-duration-s 20 --final-window-s 5 --study-name validate-fixed-current
```

A useful starting comparison is agreement within a few millivolts for voltage
and about 2% for crossover. The default is one 20-unit fixed-current run, with
both objectives averaged over the final 5 solver-clock units. Compare 10 and
20 on matching designs before selecting a shorter duration. See the
[protocol reference](fixed-current.md) for acceptance checks. The previous
165-unit sweep is available with `--evaluation-mode voltage-sweep`.
`--timeout-minutes` controls elapsed command time independently of the solver
clock. A new protocol or duration requires a fresh study name.

The temperature variable changes the feed boundaries of the parent energy
mesh and both phases. The initial temperature and collector cooling remain
313.15 K in the current template, and Arrhenius reference temperatures remain
298.15 K. Local temperature follows the energy equation rather than being
set uniformly to the sampled inlet value. Outlet backflow values also retain
the source-case settings. Assess whether the hold durations are sufficient
for thermal settling across your temperature range.

PTL bounds are dimensionless void fractions and must satisfy
`0 < min < max < 1`. `--ptl-side anode` selects only `anodeGDL` (the default);
`cathode` selects only `cathodeGDL`; `both` ties the two GDLs to the same sampled
porosity. MPL and catalyst-layer properties are not design variables.
Electronic and gas-diffusion porosities are synchronized with the flow zone.
The Darcy coefficients scale with a source-normalized Kozeny–Carman law,
preserving the source permeability at its reference porosity. This relation
is an approximation to validate for the selected material, not measured PTL
data. Pore diameter and tortuosity stay fixed; electrical contact resistance
is not newly modelled. See [the equations and limitations](architecture.md#ptl-porosity-coupling).

## Outputs and resume

Each study writes a settings-specific directory under `opt/results/` containing
`optimization.db`, `optimization_results.csv`, `pareto_front.csv`,
`knee_points.csv`, `pareto_front.png`, and `logs/trial_*`. The Pareto plot marks
the normalized Chebyshev knee and the bend-angle knee, with separate panels
coloured by thickness, water inlet temperature, and PTL porosity. All three parameters appear in
`optimization_results.csv`, `pareto_front.csv`, and `knee_points.csv`.
Each completed solver
call has its raw `trial_XXXX_solver.log`, `trial_XXXX_timeseries.csv` and
`trial_XXXX_timeseries.png`. The plot shows current, voltage and crossover,
with the objective averaging window shaded. Explicit voltage-sweep mode
instead writes `trial_XXXX_polarization_curve.csv` and its matching PNG.
Results are checkpointed after every completed evaluation.

In voltage-sweep mode, the curve CSV contains one final I-V point for each imposed voltage, both
signed and magnitude current density in A/cm2, the corresponding crossover
rate, final-window quality statistics, and flags for the points used to
interpolate the 1 A/cm2 objective.

Backfill curve CSV and PNG files for an existing voltage-sweep study without rerunning
OpenFOAM:

```bash
python3 opt/scripts/export_trial_curves.py \
  opt/results/aemec-production-20-553c273e
```

If curve CSVs already exist, the command creates only the missing PNGs. Add
`--overwrite` to regenerate both files after changing the exporter.

Overlay the stored polarization curves for the 20, 40, 60, and 80 um membrane
trials:

```bash
python3 opt/scripts/plot_membrane_polarization_curves.py \
  opt/results/aemec-production-20-553c273e
```

The comparison is written as `selected_polarization_curves.png` in the study
directory. Use `--thicknesses` to select a different set; all matching
temperatures and porosities are included. For the new continuous three-variable studies,
select completed trial IDs directly:

```bash
python3 opt/scripts/plot_membrane_polarization_curves.py \
  STUDY_DIR --trials 0 3 7
```

Curve labels include thickness, recorded inlet temperature, and recorded PTL
porosity/side. Old one- and two-variable CSVs are still supported.

Recompute knee points and update `pareto_front.png` for an existing study
without rerunning OpenFOAM:

```bash
python3 opt/scripts/find_knee_points.py \
  opt/results/aemec-production-20-553c273e
```

The command uses the two extreme normalized Pareto points for the bend-angle
references. By default, it accepts the maximum bend angle only when it is
positive. Set a stricter predefined threshold with, for example,
`--bend-angle-threshold-deg 5`. If the front contains fewer than three distinct
trade-off points, or no bend exceeds the threshold, no bend-angle knee is
reported or plotted.

Re-run the same command to resume until its completed-trial budget is reached.
Use a new `--study-name` after changing bounds, source case, operating settings,
or other study settings. `--output-dir` selects a different artifact root.
The default study name is now `aemec-thickness-temperature-porosity`. The
adapter uses objective schema 8 and rejects old one- and two-variable studies
rather than mixing different search spaces or constitutive models. Changing
`--ptl-side` also requires a new study name. Existing result files remain
available for post-processing; old stored trials are never retroactively
assigned a porosity.

To provide a serial wrapper, pass a command without shell pipes:

```bash
python3 opt/run_optimization.py --solver-command "mySolver --case-option"
```

## Troubleshooting and tests

The porosity adapter rejects inactive PTLs, mismatched source porosities
between flow/diffusion/conductivity dictionaries, unsupported conductivity
or diffusivity models, nonpositive Darcy coefficients, and nonzero
Forchheimer coefficients. It supports the source case's `DarcyForchheimer`,
`porousFSG`, and `porousSigma` models. These checks happen before writing the
PTL property updates in the scratch case.

A trial is rejected for command failures/timeouts, missing mesh files, fatal
markers or missing normal termination. Fixed-current runs also reject missing
final-window data, off-target/reversed current, or unstable voltage/crossover.
They never silently extend the configured duration or return a transient
endpoint. In explicit sweep mode, the adapter additionally checks complete
holds, a unique bracket for 1 A/cm2, and stability at interpolation endpoints.
Unused voltage holds must be complete but need not pass relative crossover
stability. Inspect the trial's logs and time series before rerunning.

The unit and mocked-adapter tests require no OpenFOAM installation:

```bash
python3 -m unittest discover -s opt -p 'test_*.py'
```
