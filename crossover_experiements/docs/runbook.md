# Crossover experiment runbook

## Prerequisites

Use a shell in which OpenFOAM is available. If the local AEMEC C++ code has
changed, build it once from the repository root before launching experiments:

```bash
./src/Allwmake
```

Install the Python plotting dependency if it is not already available:

```bash
python3 -m pip install -r crossover_experiements/requirements.txt
```

## Recommended sequence

1. Run `python3 crossover_experiements/run_experiments.py --dry-run`.
2. Inspect a configured case under `crossover_experiements/work/` if desired.
3. Run `python3 crossover_experiements/run_experiments.py`.
4. Review `summary.csv`, both PNG figures, and any failed case JSON files.

The runner continues to the next thickness after a failure. Repeating the same
command skips completed cases and retries incomplete or failed ones. Use
`--fail-fast` when debugging, or `--timeout-minutes N` to limit each mesh and
solver command.

The study manifest fingerprints the source case and records every command and
scientific setting. If those inputs change, select new work and output
directories instead of silently mixing incompatible results.

