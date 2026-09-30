# AEMEC thickness, water inlet temperature, and PTL porosity optimization

This directory contains a reusable black-box optimization package and its
AEMEC/OpenFOAM adapter. The public entry point remains:

```bash
python3 opt/run_optimization.py --iterations 50
```

Each evaluation samples **membrane thickness** (default 10–100 µm), a
**common water inlet temperature** for the anode and cathode (default
298.15–353.15 K), and **PTL porosity** (default 0.40–0.80, a dimensionless
void fraction). All bounds are configurable:

```bash
python3 opt/run_optimization.py --iterations 50 \
  --min-thickness-um 10 --max-thickness-um 100 \
  --min-water-inlet-temperature-k 298.15 \
  --max-water-inlet-temperature-k 353.15 \
  --min-ptl-porosity 0.40 --max-ptl-porosity 0.80 \
  --ptl-side anode
```

The objectives remain cell voltage and hydrogen crossover at 1 A/cm²,
interpolated from each complete voltage sweep. Temperature is applied to the
coupled parent-mesh and phase inlet boundaries; local cell temperature is
solved. Initial temperatures and collector cooling conditions remain those
of the source case, and all Arrhenius reference temperatures remain 298.15 K.
The bounds are configurable study inputs, not validated material limits.

By default the PTL variable controls **anodeGDL only**. Use `--ptl-side cathode`
for `cathodeGDL`, or `--ptl-side both` for one shared porosity applied to both
GDLs. MPL and catalyst-layer porosities stay fixed. Each selected GDL's flow,
gas-diffusivity, and electronic-conductivity porosities are synchronized.
Darcy resistance is scaled using a source-normalized Kozeny–Carman relation,
so higher porosity increases permeability while the existing `porousSigma`
model decreases effective solid conductivity. Pore diameter and tortuosity
remain fixed; no new interfacial contact-resistance model is introduced.
This is a sensitivity-study approximation requiring material-specific
validation, not a calibrated microstructure model. See the
[constitutive equations and limitations](docs/architecture.md#ptl-porosity-coupling).

All three design variables are stored in Optuna and the results, Pareto, and
knee CSVs. The Pareto figure has three panels, coloured by thickness, inlet
temperature, and porosity. New studies default to
`aemec-thickness-temperature-porosity`; existing one- or two-variable databases
cannot be resumed with this different search space. Changing `--ptl-side`
also requires a new study name. The selected side is stored in study settings
and the results CSV's `ptl_side` column.

The directory has four clear roles:

- `aemec_opt/` — importable optimizer, OpenFOAM adapter, CLI, and reporting code.
- `scripts/` — post-processing commands for completed studies.
- `tests/` — unit and case-contract tests.
- `docs/` — [architecture](docs/architecture.md) and the operational
  [runbook](docs/runbook.md).

The source case is never modified: every evaluation receives a fresh scratch
copy under `opt/work/`.

Every solver run retains its raw log, a compact per-trial polarization-curve
CSV, and a PNG plot of that curve. Existing study logs can be converted with
`opt/scripts/export_trial_curves.py` without rerunning the CFD simulations.
Use `opt/scripts/plot_membrane_polarization_curves.py STUDY_DIR --trials 0 3 7`
to overlay selected trials with all recorded design variables in the labels. Its
`--thicknesses` option includes all matching trials, including different
temperatures or porosities at the same thickness.

The standard Pareto report also writes `knee_points.csv` and marks the
normalized Chebyshev and bend-angle knee points on `pareto_front.png`. Rebuild
these artifacts for an existing study, optionally with a nonzero bend-angle
threshold, with:

```bash
python3 opt/scripts/find_knee_points.py \
  opt/results/aemec-production-20-553c273e \
  --bend-angle-threshold-deg 5
```
