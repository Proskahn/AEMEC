# AEMEC exploratory landscape analysis

Sample-based ELA for three inputs and two minimized outputs. `sample` generates
designs, `evaluate` runs them through OpenFOAM in a scratch case, and `analyze`
computes offline features. The source case is preserved. Analysis uses `pflacco`
feature implementations, not surrogate-generated observations. The exported feature set
matches the supplied table: **49 features from seven classes per landscape**.

## Install and analyze

Use Python 3.9–3.11: pflacco 1.2.2 pins NumPy 1.24 and SciPy 1.10, which
do not support Python 3.12+. From the repository root:

```bash
python3.11 -m venv ELA/.venv
ELA/.venv/bin/python -m pip install -r ELA/requirements.txt
ELA/.venv/bin/python ELA/ela.py analyze path/to/optimization_results.csv \
  --output ELA/results/study --sampling adaptive
```

The existing `opt` export works directly. Required numeric columns:

| Column | Meaning |
| --- | --- |
| membrane_thickness_um | Membrane thickness in micrometres |
| water_inlet_temperature_k | Common feed temperature in kelvin |
| ptl_porosity | Anode PTL void fraction |
| cell_voltage_v | Cell voltage at the fixed operating current |
| crossover_rate_mol_s | Anode H2 gas-source rate, mol/s |

Use one consistent case, mesh/solver protocol, PTL side (anode), and operating
current (normally 1 A/cm²) per dataset. These physical settings cannot be inferred
from the five numeric columns: the caller must ensure consistency. Optimization
exports contain successful rows only; include failed attempts if available.

Optional `status` accepts `ok`, `success`, `complete`, or `completed`
(case insensitive). Other statuses, nonfinite values, nonpositive voltage,
negative crossover, and out-of-bound inputs are excluded and preserved in
`excluded.csv` with reasons. No penalty values are substituted. Repeated successful
input designs must be aggregated explicitly before analysis; this prevents zero
distances from corrupting neighborhood features. At least 32 valid unique designs
spanning three dimensions are required; this is a guard, not a sufficiency claim.

Default bounds are 10–100 µm, 298.15–353.15 K, and 0.4–0.8. Override both with
`--lower 10 298.15 0.4 --upper 100 353.15 0.8`. Use the actual study bounds,
not observed sample extrema. These are study choices, not material limits.

On this machine, a tested Python 3.9 environment is already available at
`ELA/.venv39/bin/python`; substitute that executable in the commands above and below.

## Generate a space-filling design

```bash
ELA/.venv/bin/python ELA/ela.py sample --n 128 --seed 42 \
  --output ELA/results/design.csv
```

This writes **unevaluated** scrambled Sobol designs and a JSON sampling manifest.
Use powers of two. With identical bounds and seed, the first 32 points of a
128-point design match a 32-point design. Evaluate these exact rows using the
command below, which reuses `opt/aemec_opt/case.py`'s
`AemecOpenFoamEvaluator.evaluate()` adapter. It retains sample IDs and appends
the two objective columns and a status. It does not use adaptive TPE or replace
failed designs with new points.

Then analyze the evaluated CSV with `--sampling space-filling`. Existing adaptive
optimization data are useful exploratory evidence but do not represent uniform
coverage. The sampling flag records provenance; it cannot verify it.

## Evaluate the designs with OpenFOAM

In your Linux/container OpenFOAM environment, activate your Python environment
and run from the repository root. The extra requirements supply the existing
adapter's dependencies (including Optuna, imported by shared data types; no
optimizer is run):

```bash
python -m pip install -r ELA/requirements-evaluate.txt
command -v openFuelCell blockMesh renumberMesh topoSet splitMeshRegions
```

If executables are missing, source your OpenFOAM installation's `etc/bashrc`.
Build the repository's current solver and custom tools with `./Allwmake` from
the `src` directory if not already built. The adapter expects the current
hydrogen-crossover logging in this repository's solver.

Start with **one design** to check the CFD setup:

```bash
python ELA/ela.py evaluate ELA/results/design.csv \
  --output-dir ELA/results/evaluation --limit 1
```

This runs the first eligible row, remeshing a fresh copy of `run/AEMEC`, applying
its membrane thickness, common water inlet temperature and anode PTL porosity,
then running `make mesh` and `openFuelCell`. Every evaluation uses the full
voltage sweep in the source case and interpolates both objectives at
**10,000 A/m² = 1 A/cm²**, with the same settling/bracketing checks as `opt`.

Continue with all remaining designs:

```bash
python ELA/ela.py evaluate ELA/results/design.csv \
  --output-dir ELA/results/evaluation
```

Repeating this command skips completed and failed rows, and resumes pending or
interrupted rows. `--limit` caps attempts **in this invocation**, not the total
number of successful evaluations. Ctrl-C records the interrupted attempt; after
a hard process termination a `running` row is also eligible to run again.
After investigating failed cases, explicitly retry them with:

```bash
python ELA/ela.py evaluate ELA/results/design.csv \
  --output-dir ELA/results/evaluation --retry-failed
```

Results are checkpointed after every attempt:

- `evaluated.csv`: all design rows, including pending/failed rows with blank
  objectives, statuses, failure reasons, attempt counts, and log paths.
- `checkpoint.json`: authoritative resume state, design hash, case/adapter hashes,
  commands, and operating settings. A stale CSV is rebuilt from this state.
- `logs/sample_NNNN/attempt_NNN/`: per-attempt mesh and solver logs, plus
  polarization CSV/PNG when available. Retries do not overwrite earlier logs.

Runs are sequential and use a dedicated scratch directory under `ELA/work`.
The scratch case is replaced between evaluations; full CFD fields for earlier
designs are not retained. Locks prevent simultaneous runs from using the same
output or scratch directory. Use a fresh output directory when changing the
design, source case, adapter, or simulation settings. Keep your compiled solver
consistent too; the checkpoint does not fingerprint compiled binaries.

Useful options (`python ELA/ela.py evaluate --help` lists all):

- `--dry-run`: validates design/resume settings and prints the planned count and
  paths without writing output or executing CFD; does not check executable availability.
- `--case PATH`, `--work-dir PATH`: source case and dedicated scratch root.
- `--timeout-minutes N`: timeout for **each** mesh or solver command.
- `--mesh-command "make mesh"`, `--solver-command "openFuelCell"`: command
  overrides, parsed as arguments without shell expansion. Defaults are serial.
- `--max-consecutive-failures 3`: stop after repeated failures, preserving the
  remaining pending designs. Remaining failures produce a nonzero exit status.
- `--target-current-density-a-m2`, `--ptl-side`, and stability options: defaults
  match the adapter. Changing them requires a new output directory.

Finally, compute the normalized ELA features:

```bash
python ELA/ela.py analyze ELA/results/evaluation/evaluated.csv \
  --output ELA/results/analysis --sampling space-filling
```

Analysis requires at least 32 successful unique designs. Pending/failed designs
remain in the audit output; they do not become objective penalties. For custom
sampling bounds, supply the same `--lower` and `--upper` bounds to `analyze`.

## Features and outputs

Inputs are mapped to [0,1] using the supplied domain. Each objective is affinely
scaled by its sample minimum and maximum. Three additional landscapes use
`w * normalized_voltage + (1-w) * normalized_crossover`, for w = 0.25, 0.5, 0.75.
The standalone outputs supply the w = 1 and w = 0 cases.

For comparable analyses across sample sizes or seeds, fix objective scales:

```bash
ELA/.venv/bin/python ELA/ela.py analyze path/to/evaluated.csv \
  --output ELA/results/fixed-scales --sampling space-filling \
  --objective-scales 1.3 2.3 0 1e-7
```

Those example endpoints are illustrative, not prescribed physical limits.
Values outside objective scaling endpoints are not clipped. Input bounds still
apply. A constant objective maps to zero with sample-derived scaling and its
standalone feature families are skipped.

The seven classes and exact selected features are below. Braces expand to all
listed suffixes; exported names retain the `pflacco` convention.

| Class | Count | Exported features |
| --- | ---: | --- |
| y-distribution | 3 | `ela_distr.{skewness,kurtosis,number_of_peaks}` |
| Level set | 9 | `ela_level.{mmce_lda,mmce_qda,lda_qda}_{10,25,50}` |
| Meta-model | 9 | `ela_meta.lin_simple.{adj_r2,intercept}`, `ela_meta.lin_simple.coef.{min,max,max_by_min}`, `ela_meta.lin_w_interact.adj_r2`, `ela_meta.quad_simple.{adj_r2,cond}`, `ela_meta.quad_w_interact.adj_r2` |
| Dispersion | 16 | `disp.{ratio_mean,ratio_median,diff_mean,diff_median}_{02,05,10,25}` |
| NBC | 5 | `nbc.nn_nb.{sd_ratio,mean_ratio,cor}`, `nbc.dist_ratio.coeff_var`, `nbc.nb_fitness.cor` |
| PCA | 2 | `pca.expl_var_PC1.{cov_init,cor_init}` |
| ICoFiS | 5 | `ic.{h_max,eps_s,eps_max,eps_ratio,m0}` |

The figure's `ic.h.max` and `ic.eps.{s,max,ratio}` correspond to the library's
`ic.h_max` and `ic.eps_{s,max,ratio}`. NBC uses the library spelling
`nbc.nb_fitness.cor`. Runtime counters and the other six PCA features are excluded.
All classes use existing observations only. Convexity, finite-difference curvature
and local-search routines requiring new simulator calls remain absent.

Level sets use thresholds at the 10%, 25%, and 50% output quantiles, with 3-fold
stratified LDA/QDA classification as before. Dispersion now includes 2% and 5%
subsets as well as 10% and 25%. Following pflacco, a subset includes **all values
at or below the quantile**, including ties. A subset with fewer than two points
cannot supply pairwise distances: its four features remain blank, other quantiles
are still computed, and subset sizes are recorded in the diagnostics. Even when
defined, estimates from very small subsets can be unstable.

PCA uses the augmented matrix `[normalized X, normalized y]` for each landscape.
The two reported values are the first principal component's explained variance
fraction for the covariance and correlation versions of that matrix.

Information content follows the ICoFiS implementation in pflacco, accounting for
objective differences divided by input-space step distances. By default it uses a
nearest-neighbor tour (`--ic-sorting nn`) with a starting point selected using
`--seed`. For a randomly permuted tour of the same samples, use
`--ic-sorting random`. These are tours of existing observations, not additional
CFD evaluations or a newly simulated continuous random walk. The chosen tour,
seed, epsilon grid, and sensitivity thresholds are recorded in `metadata.json`.
The epsilon grid is zero plus 1,000 logarithmically spaced values from 1e-5 to 1e15;
settling and partial-information sensitivities are 0.05 and 0.5, respectively.
In pflacco 1.2.2, `eps_s` and `eps_ratio` are **log10 epsilon** values; `eps_max`
is returned on the **linear epsilon** scale. Nonfinite values remain blank.

Every landscape always has 49 named rows, including when a family fails or an
output is constant. No missing feature is imputed to zero. Constant outputs skip
all families and retain 49 blank values, including PCA, as an explicit pipeline
policy rather than a claim that every individual feature is undefined.

- `features.csv`: long-form landscape / feature / value table; 245 rows for the
  two objectives and three weighted combinations; undefined values blank.
- `metadata.json`: input hash, versions, normalization, counts, seed, per-family
  warnings/errors, quantile subset sizes and undefined-feature names. Includes the
  full 49-feature catalog and computation settings. A `partial` status means some
  family values are undefined; `failed` means the call failed and all its values
  are blank. Family errors retain other results.
- `accepted.csv`, `excluded.csv`: audited rows and preserved original metadata.
- `pareto.csv`, `pareto.png`: sampled nondominated points; black rings mark them.
- `spearman.csv`: input/output rank correlations, not causal sensitivity indices.
- `report.md`: counts and interpretation limitations.

Meta-model fit describes global trends; it is not held-out predictive accuracy.
Output-density peaks are not a count of local optima. Nearest-better and dispersion
features suggest spatial organization, not proof of basin structure. Weighted
sums do not capture all Pareto geometry. Compare independent sampling seeds and
nested sample sizes; this version does not compute confidence intervals or
surrogate response surfaces. Inspect numerical settling and mesh sensitivity
separately before interpreting roughness as physical behavior.

## Verification

```bash
ELA/.venv/bin/python -m unittest discover -s ELA/tests -v
```

Install `requirements-evaluate.txt` to run the full suite, including mock-based
batch/resume tests. Tests do not launch OpenFOAM.

References: [Mersmann et al. (2011)](https://doi.org/10.1145/2001576.2001690),
[pflacco API](https://pflacco.readthedocs.io/en/latest/pflacco.classical_ela_features.html)
(including the ICoFiS implementation and its Muñoz, Kirley and Halgamuge reference).
