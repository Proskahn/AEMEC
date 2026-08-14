# AEMEC gas-crossover experiments

This directory is the end-to-end workflow for the fixed-current membrane
thickness study. It compares **20, 40, 60, and 80 µm** membranes at
**1 A/cm² for 20 s**, starting the galvanostatic controller from 2 V.

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
