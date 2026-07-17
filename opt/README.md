# AEMEC membrane-thickness optimization

This directory composes a reusable black-box optimization library with an
AEMEC/OpenFOAM case adapter. The public entry point remains:

```bash
python3 opt/run_optimization.py --iterations 50
```

Documentation is separated by concern:

- [Architecture](docs/architecture.md) — module boundaries and extension points.
- [Optimizer library](docs/optimizer.md) — model-independent search, persistence,
  failure handling, and Pareto selection.
- [AEMEC OpenFOAM adapter](docs/aemec-openfoam.md) — geometry, voltage-sweep
  interpolation, and objective validation.
- [Runbook](docs/runbook.md) — setup, commands, outputs, validation, and
  troubleshooting.

The source case is never modified: every evaluation receives a fresh scratch
copy under `opt/work/`.
