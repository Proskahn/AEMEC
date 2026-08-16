# AEMEC 165 s polarization curve

This workflow runs the former potentiostatic AEMEC polarization schedule in
an isolated work case. It does not modify `run/AEMEC` and does not enable the
new galvanostatic crossover controller.

The retained schedule applies eleven 15 s voltage holds from 1.3 to 2.3 V,
for a total simulated time of 165 s. After a normal solver exit, the workflow
uses the maintained modules in `visualization/` to generate the polarization
curve, voltage decomposition, and hydrogen-crossover plots.

## Run

Source the OpenFOAM environment, then run from the repository root:

```bash
python3 polarization_curve/run_polarization_curve.py
```

The normal command always performs a fresh solve. It replaces only a previous
result directory carrying this workflow's ownership manifest, copies the
source case without old time directories or logs, remeshes, and runs
`openFuelCell`. OpenFOAM output is written to `logs/solver.log` and displayed
live in the terminal, including each `Time = ...` line.

Inspect the configured scratch case without running OpenFOAM:

```bash
python3 polarization_curve/run_polarization_curve.py --dry-run
```

Refuse to replace an existing result directory:

```bash
python3 polarization_curve/run_polarization_curve.py --keep-existing
```

If OpenFOAM completed but plotting was interrupted, regenerate the products
without rerunning the solver:

```bash
python3 polarization_curve/run_polarization_curve.py --postprocess-only
```

## Output layout

```text
polarization_curve/results/former-165s-voltage-sweep/
├── run.json
├── logs/
│   ├── mesh.log
│   └── solver.log
├── data/
│   ├── polarization_curve.csv
│   ├── voltage_decomposition.csv
│   └── hydrogen_crossover.csv
└── figures/
    ├── polarization_curve.png
    ├── voltage_decomposition.png
    └── hydrogen_crossover.png
```

The isolated OpenFOAM case is written to
`polarization_curve/work/former-165s-voltage-sweep/case`.
