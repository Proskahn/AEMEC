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

Older controller binaries started the first target's hold timer at the first
0.1 s update rather than at the case start. Consequently, a 20 s minimum hold
could report `accepted: false` at the 20 s cutoff even with `stability samples:
5/5`. The runner now recognizes that exact legacy condition and reconstructs
the CSV, summary, and plots directly from the completed solver log. A failed
case with fewer than 5/5 final stable samples is not recovered because its
endpoint is genuinely unsettled.

## Current-density sweep

Run the second experiment independently of the thickness study:

```bash
python3 crossover_experiements/run_current_sweep.py --dry-run
python3 crossover_experiements/run_current_sweep.py
```

It uses a 30 µm membrane, 8.0e-5 m² active area, 0.2 A/cm² current increments,
20 s minimum holds, and a 400 s safety limit. Completed runs resume from their
CSV data; a normally ended log containing all accepted targets can also be
recovered automatically. If a target is not accepted before the safety limit,
`case.json` identifies it and the partial flux time series remains available.

Use a different output/work pair whenever changing the step, thickness, area,
hold duration, or maximum duration. This prevents incompatible sweep data from
being combined.

The study manifest fingerprints the source case and records every command and
scientific setting. If those inputs change, select new work and output
directories instead of silently mixing incompatible results.
