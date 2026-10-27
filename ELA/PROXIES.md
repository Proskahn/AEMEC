# ELA-matched proxy functions

`generate_proxies.py` implements the **proxy generation** part of Yin et al.,
[Landscape-aware Automated Algorithm Design: An Efficient Framework for Real-world
Optimization](https://arxiv.org/abs/2602.04529v1), sections 3.1–3.2, Algorithm 1,
Table 1, and section 5.3. Here GP means **genetic programming**, not Gaussian
process regression. The LLM algorithm-discovery stage is outside this tool.

The generator evolves cheap mathematical functions whose ELA characteristics
resemble your measured AEMEC landscape. Fitness uses ELA similarity, not error
between the function's predictions and the measured responses. A generated
function is a synthetic optimization benchmark, not a voltage/crossover predictor.

## Run on the existing analysis

Use the already configured Python 3.9 environment on this Mac. No new packages
are needed beyond `ELA/requirements.txt`. The newer `ELA/.venv` was created with
Python 3.13 and is incompatible with these pinned ELA dependencies.

Start with a modest search to check the workflow:

```bash
ELA/.venv39/bin/python ELA/generate_proxies.py fit ELA/results/analysis \
  --output ELA/results/proxies_quick \
  --population 12 --generations 3 --top-k 3 --seed 42
```

Both objectives get their own search. To use the paper's search settings:

```bash
ELA/.venv39/bin/python ELA/generate_proxies.py fit ELA/results/analysis \
  --output ELA/results/proxies_paper \
  --population 50 --generations 50 --repeats 5 --subset-fraction 0.8 \
  --correlation-threshold 0.9 --crossover 0.5 --mutation 0.1 \
  --min-depth 3 --max-depth 12 --top-k 3 --seed 42
```

The input folder must contain `features.csv`, `accepted.csv`, and `metadata.json`
from `ela.py analyze`. **One previously computed feature vector is not enough
to recover feature distributions.** The generator verifies that the full-pool
features agree with the accepted observations, then computes the repeated-subset
features from these existing observations. It never calls OpenFOAM.

The paper uses 150 × dimension points: **450 for your three-dimensional space**.
The generator uses however many valid samples are already available, and records
the difference from the paper's sample count. Your 128-point pool can be used;
it is not silently expanded or replaced with model predictions. At the default
80% fraction, at least 40 accepted designs are needed for 32-point subsets.

Each candidate needs five ELA computations. A 50 × 50 search can take substantial
time, even though evaluating the expression itself is cheap. Progress is printed
per generation and saved to `history.csv`; `best_so_far.json` is a usable checkpoint
after each generation. Ctrl-C leaves completed checkpoints and an interrupted
manifest. Resume is not implemented; use a new output directory for a fresh run.

## Method

1. Recover normalized inputs and objectives using the analysis metadata. Input
   bounds remain fixed; objective scaling matches the original ELA analysis.
2. Select five independent random subsets, each containing 80% of the original
   designs without replacement. Compute the same 49 features on each subset.
3. Remove target features undefined on any repeat. Greedily prune features with
   absolute Pearson correlation greater than 0.9, using the existing feature order.
   Finite constant features remain as matching constraints. Freeze the selection.
4. Initialize expression trees by alternating full and grow construction with
   random depths 3–12. Evaluate each expression on the original design pool.
5. Reject nonfinite/constant outputs. Normalize each candidate's output by its
   full-pool minimum and range. Compute its ELA distributions on the **same**
   subsets and with the same ELA random seed as the target.
6. Minimize mean per-feature Wasserstein distance. Select parents by tournament,
   apply one-point subtree crossover and subtree mutation, and retain a separate
   archive of the best expressions encountered.
7. Export the top three distinct expressions, all 49 full-pool feature comparisons,
   and distances on five new random subsets of the same sample pool.

With retained features J, the explicitly implemented similarity is

```text
distance = mean over j in J of W1(target[:, j], candidate[:, j]) / scale[j]
```

The empirical distributions are over subset estimates of **the same feature**.
Skewness, regression coefficients, and other unrelated features are never flattened
into one interchangeable set of observations. The target selection and scales
stay fixed; a missing selected candidate feature incurs infinite fitness.

## Primitive definitions and implementation choices

The paper does not specify every implementation detail, so this is a documented
implementation of its method, not a claim of bit-for-bit reproduction of its DEAP
code. This version implements the evolutionary loop directly using NumPy.

The 20 operators from Table 1 are:

| Group | Operators |
| --- | --- |
| Binary, with scalar/vector broadcasting | `add`, `sub`, `mul`, `div` |
| Elementwise unary | `neg`, `rec`, `mult10`, `square`, `abs`, `sqrt`, `exp`, `ln`, `sin`, `cos`, `round` |
| Vector operations | `sum`, `mean`, `cum`, `prod`, `max` |

`sqrt(a)` means sqrt(abs(a)); `ln(a)` means log(abs(a)); sin/cos use a 2π
factor. `round` uses NumPy's nearest-even rounding. `rec` means 1/a and
`cum` is a cumulative sum along design coordinates. Vector reductions operate
within one design, never across the batch of sample rows.

Terminals are the normalized design vector `u`, index vector `[1, 2, 3]`, and real
constants. Real constants are chosen from {-2, -1, 0.5, 1, 2} or sampled uniformly
from [-5, 5]. A `rand` constant is sampled **once when creating its node**, so the
exported proxy is a deterministic function. A vector-valued root is reduced by
summing its coordinates to obtain one scalar objective.

Other explicit choices:

- Feature distances use raw feature units by default. `--distance-scaling target-std`
  is an optional extension: divide by target repeat standard deviations, floored
  at `1e-6 × max(1, abs(target mean))`. This changes the optimization criterion.
  The paper does not provide an unambiguous component aggregation/scaling formula.
- Existing ELA input/output normalization, NN information-content ordering, and
  other feature settings are inherited from the analysis. Candidate normalization
  is fixed over the full pool, not independently recomputed for every subset.
- Tournament size is 3. `--generations` counts offspring generations after the
  initialized population (generation 0). There is a best-ever archive but no
  forced population elitism. Top functions are structurally distinct; they are
  not guaranteed to represent genuinely different landscapes.
- The default tree cap is 255 nodes in addition to the depth cap of 12. Invalid
  variation is replaced by its parent. Mutation generates replacement subtrees
  of depth 0–3 within the remaining depth allowance. All are recorded choices
  where the paper leaves settings unspecified.
- Invalid arithmetic is penalized, not silently clipped or imputed. We additionally
  check a seeded 256-point Sobol pool, all domain corners, and the center for
  finite output. This is a numerical safeguard, not a proof of validity everywhere.
- The original sample pool is reused, without expensive simulator evaluations.

## Inspect and use a generated proxy

At the top level, `summary.csv` lists training/validation distances and expressions;
`manifest.json` records settings, dependency versions, input hashes, and exact subsets.
Each landscape contains feature selection, target distributions, and search history.
Each `proxy_01`, `proxy_02`, ... folder contains:

- `proxy.json`: the whitelisted expression tree and fixed input/output transforms.
- `proxy.py` and `proxy_runtime.py`: portable callable, requiring only NumPy.
- `feature_comparison.csv`: all 49 target and generated features, selected-feature
  indicators, and Wasserstein components used in fitness.
- `feature_distributions.csv` and `validation_feature_distributions.csv`.
- `validation_diagnostics.json`: selected features undefined on validation repeats,
  distinguishing failures in the target from failures in the generated proxy.
- `sample_values.csv`: original physical inputs, target normalized values, and
  generated values for inspection. These are not additional CFD observations.

Evaluate physical input rows from an existing design CSV:

```bash
ELA/.venv39/bin/python ELA/generate_proxies.py evaluate \
  ELA/results/proxies_quick/cell_voltage_v/proxy_01/proxy.json \
  --input ELA/results/analysis/accepted.csv \
  --output ELA/results/proxy_values.csv
```

Or call directly from Python, from the repository root:

```python
from ELA.proxy_functions import ProxyFunction

proxy = ProxyFunction('ELA/results/proxies_quick/cell_voltage_v/proxy_01/proxy.json')
value = proxy([50.0, 333.0, 0.6])  # µm, K, dimensionless porosity
```

Alternatively, copy an entire `proxy_01` folder elsewhere and import `objective`
from its `proxy.py`. It is callable on either one design or an (n, 3) array.
Outputs are **dimensionless proxy values**, not volts or mol/s. The normalization
uses the search sample pool, so values at new points need not lie in [0, 1].

`--landscapes cell_voltage_v` restricts generation to voltage. The existing
`weighted_voltage_0.25`, `weighted_voltage_0.5`, and `weighted_voltage_0.75`
landscapes can also be selected if present in the analysis.

## Interpretation

Smaller matching distance is better within a run, but no universal threshold
establishes sufficient fidelity. Inspect the full feature comparison, not only
the optimized subset, and compare several seeds/budgets. Five resamples are a
small basis for correlation pruning, as in the paper. The validation score uses
new subsets of the **same** evaluated pool; it is a resampling robustness check,
not an independent test on unseen CFD data. Undefined validation scores remain blank.
For example, `ela_level.lda_qda_50` can be undefined when the classifiers both have
zero error on a subset. A target-side undefined feature prevents an aggregate
validation score even if the proxy itself is finite. The feature mask is kept
fixed and the affected features/repeats are recorded explicitly.

The paper's method concerns scalar objectives. Generating a voltage-like and a
crossover-like proxy independently does **not** preserve their contradictory
dependence on membrane thickness, pointwise rankings, minimizers, or Pareto front.
Do not treat them as a validated two-objective surrogate for AEMEC. The functions
are intended for low-cost optimizer experiments followed by validation on the
actual problem. LLM-driven optimizer generation is not part of this implementation.
