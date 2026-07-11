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

The default 50-evaluation study searches 10–100 um:

```bash
python3 opt/run_optimization.py \
  --iterations 50 \
  --min-thickness-um 10 \
  --max-thickness-um 100
```

The bounds are engineering inputs, not solver-inferred material limits. Use a
smaller pilot before a full study:

```bash
python3 opt/run_optimization.py \
  --iterations 20 --startup-trials 6 \
  --study-name aemec-fast-pilot
```

Validate fast and ramp modes at thin, middle, and thick designs before a long
run:

```bash
python3 opt/run_optimization.py --iterations 3 --startup-trials 3 --study-name validate-fast
python3 opt/run_optimization.py --iterations 3 --startup-trials 3 --run-mode ramp --study-name validate-ramp
```

A useful starting comparison is agreement within a few millivolts for voltage
and about 2% for crossover. Increase `--solver-iterations` or use ramp mode if
fast mode is not stable. Use `--current-relative-tolerance` and
`--timeout-minutes` to adjust acceptance limits and command timeout.

## Outputs and resume

Each study writes a settings-specific directory under `opt/results/` containing
`optimization.db`, `optimization_results.csv`, `pareto_front.csv`,
`pareto_front.png`, and `logs/trial_*`. Results are checkpointed after every
completed evaluation.

Re-run the same command to resume until its completed-trial budget is reached.
Use a new `--study-name` after changing bounds, source case, operating settings,
or other study settings. `--output-dir` selects a different artifact root.

To provide a serial wrapper, pass a command without shell pipes:

```bash
python3 opt/run_optimization.py --solver-command "mySolver --case-option"
```

## Troubleshooting and tests

A trial is rejected for command failures/timeouts, missing mesh files, OpenFOAM
fatal markers, missing normal termination, missing target data, or failed
operating-point/stability checks. Inspect that trial's log directory, correct
the cause, and rerun the same command.

The unit and mocked-adapter tests require no OpenFOAM installation:

```bash
python3 -m unittest discover -s opt -p 'test_*.py'
```
