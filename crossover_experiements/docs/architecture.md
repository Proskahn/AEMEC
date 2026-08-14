# Workflow architecture

The workflow separates the source case, disposable run cases, and durable
results:

```text
run/AEMEC (read only)
       │ clean copy + membrane-thickness rewrite
       ▼
crossover_experiements/work/<study>/<thickness>um
       │ make mesh → openFuelCell → log parser
       ▼
crossover_experiements/results/<study>
       ├── logs and case metadata
       ├── per-thickness time-series CSV
       └── summary CSV and comparison figures
```

`crossover_experiments/config.py` owns the scientific definition. It changes
only the membrane geometry and the experiment controls: galvanostatic mode,
the signed solver target of −10,000 A/m², 20 s duration, 0.1 s step, and 2 V
initial potential. The inactive voltage sweep remains in the copied case so
the original optimization workflow is unaffected.

`runner.py` owns clean copying, external commands, safe resume, run validation,
and artifact paths. A case is marked completed only when the solver ends
normally, reaches 20 s, and its final-window current is within 1% of the target.

`parsing.py` turns solver diagnostics into explicit scientific quantities. The
hydrogen production reference is calculated from Faraday's law,
`n_H2 = |I|/(2F)`, and the crossover fraction is
`100 n_cross/n_H2`. `reporting.py` reads only these normalized records, which
makes plotting independent of OpenFOAM.

