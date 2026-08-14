# AEMEC gas-crossover experiments

This directory contains two end-to-end gas-crossover studies:

1. A membrane-thickness comparison at **1 A/cm² for 20 s** using 20, 40,
   60, and 80 µm membranes.
2. A current-density continuation sweep from **0 to 2 A/cm²** in 0.2 A/cm²
   increments using the baseline 30 µm membrane.

The source case under `run/AEMEC` is never modified by this workflow. Every
thickness receives a clean OpenFOAM case under `crossover_experiements/work/`.
The runner can resume completed cases and keeps the raw mesh and solver logs,
parsed time-series CSV files, per-case metadata, a combined summary, and plots.
It also recovers a normally completed solver log from an older controller
binary without rerunning the CFD case when the final record contains the full
stable-sample window.

## Run the study

From the repository root, first load your OpenFOAM environment and build the
AEMEC libraries/solver if necessary. Then run:

```bash
python3 crossover_experiements/run_experiments.py
```

Useful checks and controls:

```bash
# Configure all four clean cases without meshing or solving
python3 crossover_experiements/run_experiments.py --dry-run

# Resume an interrupted study (completed thicknesses are skipped)
python3 crossover_experiements/run_experiments.py

# Deliberately rerun and replace every generated case result
python3 crossover_experiements/run_experiments.py --overwrite

# Run only two thicknesses in a separate study directory
python3 crossover_experiements/run_experiments.py \
  --thicknesses 20 60 \
  --work-dir crossover_experiements/work/20um-60um \
  --output-dir crossover_experiements/results/20um-60um
```

The default output directory is
`crossover_experiements/results/fixed-current-1Acm2-20s/`. Its contents are:

```text
experiment.json             reproducibility manifest
cases/*.json                status and final-window summary for each thickness
logs/*_mesh.log             raw meshing logs
logs/*_solver.log           raw openFuelCell logs
data/*_timeseries.csv       current, voltage, crossover, and controller state
summary.csv                 final-5-s comparison
timeseries.png              four transient comparison panels
thickness_summary.png       crossover, crossover fraction, and voltage vs thickness
```

Regenerate the summary and plots without rerunning OpenFOAM:

```bash
python3 crossover_experiements/scripts/plot_results.py
```

See [the runbook](docs/runbook.md) for failure recovery and
[the architecture note](docs/architecture.md) for the data flow.

## Current-density crossover sweep

The second experiment holds each of 11 current targets for at least 20 s and
allows up to 400 s for the complete continuation sweep. Run it with:

```bash
# Configure and inspect without solving
python3 crossover_experiements/run_current_sweep.py --dry-run

# Run the complete 0--2 A/cm2 sweep
python3 crossover_experiements/run_current_sweep.py
```

The current step is configurable—for example, `--current-step-a-cm2 0.1`
requests 21 targets. The step must divide 2 A/cm² exactly.

The mesh has a membrane area of

```text
(40 mm)(2 mm) = 80 mm² = 8.0e-5 m² = 0.8 cm².
```

The reported area-averaged hydrogen crossover flux density is

```text
N_H2,cross = n_dot_H2,cross / A_membrane    [mol/(m² s)].
```

The default output directory is
`crossover_experiements/results/current-sweep-0-to-2Acm2/` and contains:

```text
current_sweep.csv             accepted-point flux, voltage, and current data
data/timeseries.csv           transient rate and flux-density history
current_sweep.png             flux, voltage, and crossover fraction vs current
current_sweep_timeseries.png  target/measured current and flux vs time
logs/solver.log               complete OpenFOAM diagnostics
```

Regenerate both figures from the saved CSV without rerunning OpenFOAM:

```bash
python3 crossover_experiements/scripts/plot_current_sweep.py
```
