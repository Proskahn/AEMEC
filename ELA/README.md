# AEMEC exploratory landscape analysis

Offline, sample-based ELA for three inputs and two minimized outputs. This code
never launches OpenFOAM or changes the source case. It uses `pflacco` feature
implementations, not surrogate-generated observations. The exported feature set
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
128-point design match a 32-point design. Evaluate these exact rows through the
existing `opt/aemec_opt/case.py` `AemecOpenFoamEvaluator.evaluate()` adapter, retaining
sample IDs and appending the two objective columns and a status. The adapter takes
`trial_number, thickness_um, water_inlet_temperature_k, ptl_porosity`. This folder
does not yet provide a CFD batch runner. Do not feed the design through adaptive
TPE, which would choose different locations.

Then analyze the evaluated CSV with `--sampling space-filling`. Existing adaptive
optimization data are useful exploratory evidence but do not represent uniform
coverage. The sampling flag records provenance; it cannot verify it.

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

References: [Mersmann et al. (2011)](https://doi.org/10.1145/2001576.2001690),
[pflacco API](https://pflacco.readthedocs.io/en/latest/pflacco.classical_ela_features.html)
(including the ICoFiS implementation and its Muñoz, Kirley and Halgamuge reference).
