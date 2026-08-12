# AEMEC visualization tools

This directory contains source code only. Generated CSV and PNG files are
written to `visualization/output/`, which is intentionally ignored by Git.

## Tools

- `polarization_curve.py` extracts accepted polarization points and plots the
  current-density/voltage curve.
- `aemec_diagnostics.py` plots voltage decomposition and dissolved-H2/gas
  crossover diagnostics for the same accepted points.
- `pareto_trials.py` plots a selected prefix of an optimization-results CSV.

Run the standard plots from the repository root:

```bash
python3 visualization/polarization_curve.py \
    --log run/AEMEC/log.run \
    --scan-mode voltage \
    --hold-duration 15

python3 visualization/aemec_diagnostics.py \
    --log run/AEMEC/log.run \
    --scan-mode voltage
```

Plot optimization trials with:

```bash
python3 visualization/pareto_trials.py opt/results/pareto_front.csv
```

Run the focused tests with:

```bash
PYTHONPATH=. python3 -m unittest discover \
    -s visualization/tests -p 'test_*.py'
```
