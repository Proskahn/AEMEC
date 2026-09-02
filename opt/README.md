# AEMEC membrane-thickness optimization

This directory contains a reusable black-box optimization package and its
AEMEC/OpenFOAM adapter. The public entry point remains:

```bash
python3 opt/run_optimization.py --iterations 50
```

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
Use `opt/scripts/plot_membrane_polarization_curves.py` to overlay polarization
curves for selected membrane thicknesses.

The standard Pareto report also writes `knee_points.csv` and marks the
normalized Chebyshev and bend-angle knee points on `pareto_front.png`. Rebuild
these artifacts for an existing study, optionally with a nonzero bend-angle
threshold, with:

```bash
python3 opt/scripts/find_knee_points.py \
  opt/results/aemec-production-20-553c273e \
  --bend-angle-threshold-deg 5
```
