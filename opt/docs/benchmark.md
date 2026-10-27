# Benchmark GP BO, TPE BO, NSGA-II, and CMA-ES

`opt/run_benchmark.py` compares four algorithms on the same three-variable,
two-objective minimization problem. It leaves `run_optimization.py` and existing
studies unchanged. Inputs remain membrane thickness, water inlet temperature,
and anode PTL porosity by default.

| Algorithm key | Implementation |
| --- | --- |
| `gp` | Two independent Matérn-5/2 Gaussian processes with ARD; analytic expected hypervolume improvement (EHVI) |
| `tpe` | Optuna multivariate multiobjective TPE (MOTPE), 64 EI candidates |
| `nsga2` | Optuna NSGA-II, simulated binary crossover (eta=20, probability 0.9) and per-variable random-reset mutation (probability 1/3) |
| `cmaes` | DEAP multiobjective CMA-ES (MO-CMA-ES), one offspring per update, Pareto-rank and fixed-reference hypervolume selection |

NSGA-II is the standard name for the requested “NSAG-II”. Its population size
equals `--initial`. This is Optuna's NSGA-II variant; its mutation resets a
selected variable, rather than using polynomial mutation.
The benchmark requires Optuna 4.9; queued initial observations are explicitly
assigned to population 0 using its generation API, so they participate in evolution.

All four algorithms run by default. Select a subset with `--algorithms`; for
example, `--algorithms gp tpe nsga2` reproduces the previous three-method setup.

## Run on your generated ELA proxies

From the repository root, install the dependencies into your Python environment.
The benchmark now includes DEAP for MO-CMA-ES. On this Mac:

```bash
ELA/.venv39/bin/python -m pip install -r opt/requirements-benchmark.txt
```

Then run all four algorithms:

```bash
ELA/.venv39/bin/python opt/run_benchmark.py \
  --backend proxies --proxy-root ELA/results/proxies_verified --proxy-rank 1 \
  --output opt/results/benchmark_proxies \
  --budget 64 --initial 16 --seeds 42 43 44 45 46
```

This pairs `cell_voltage_v/proxy_01/proxy.json` and
`crossover_rate_mol_s/proxy_01/proxy.json`. They are evaluated as deterministic,
dimensionless objectives named `voltage_proxy` and `crossover_proxy`. No ELA
features are recomputed and no CFD runs are launched. Select ranks 2 or 3 with
`--proxy-rank`, or point `--proxy-root` to another generation run. Reuse the same
pair across all algorithms in a comparison, and use a new output directory per pair.

**These independent scalar proxies have an artificial joint Pareto front.**
Their separate ELA matches do not establish a faithful AEMEC trade-off. A proxy
benchmark tests the algorithm workflow; rankings must be validated on AEMEC.

A faster smoke run is:

```bash
ELA/.venv39/bin/python opt/run_benchmark.py \
  --backend proxies --proxy-root ELA/results/proxies_verified \
  --output opt/results/benchmark_quick \
  --budget 24 --initial 8 --seeds 42 43 \
  --gp-candidates 256 --gp-restarts 0
```

For an independent analytic check, use `--backend zdt1`. This uses the standard
three-dimensional ZDT1 function, mapping the physical input bounds onto [0,1]^3.
It is explicitly synthetic and does not use the AEMEC model or proxy files.

To run **only CMA-ES** on the proxy pair:

```bash
ELA/.venv39/bin/python opt/run_benchmark.py \
  --backend proxies --proxy-root ELA/results/proxies_verified \
  --algorithms cmaes --cma-sigma 0.2 \
  --output opt/results/benchmark_cmaes \
  --budget 64 --initial 16 --seeds 42 43 44 45 46
```

## How CMA-ES handles the two objectives

The `cmaes` option uses **multiobjective CMA-ES**, implemented by DEAP's
`StrategyMultiObjective` (Voss, Hansen and Igel, 2010). Voltage and crossover
remain separate objectives; no scalar weighting is introduced. This is an
evolution strategy, not a Bayesian optimization model.

Each parent design has a Gaussian search distribution. Its covariance learns
promising combinations of the three input variables, while its step size learns
how far to search. A new offspring competes with the parents: Pareto ranking
determines priority, and the least hypervolume contributor is removed when a
front does not fit. The adapter uses the benchmark's **fixed reference** for this
truncation, replacing DEAP's default population-dependent reference.

The implementation choices are:

- **Initialization:** successful shared Sobol observations become parents.
  The parent count equals `--initial` when all succeed. Initial failures reduce
  it; there are no free replacement evaluations. If all fail, a logged Sobol
  fallback continues until its first success, then initializes one parent.
- **Sequential updates:** one offspring is evaluated and the strategy updates
  before the next proposal (`lambda=1`). The exact budget applies even when it
  is not a multiple of a population size.
- **Initial scale:** `--cma-sigma 0.2` sets a standard deviation of 20% of each
  input range, initially with identity covariance. For the default physical
  bounds, this is 18 µm, 11 K, and 0.08 porosity before boundary handling.
  The parameter must be finite and positive; it subsequently adapts per parent.
- **Bounds:** reflect proposals into the unit box, e.g. -0.2 → 0.2 and 1.2 → 0.8.
  The repaired coordinates are used for both evaluation and adaptation.
  No boundary-resampling evaluations are hidden from the budget.
- **Failed offspring:** charge the attempt and skip selection/adaptation.
  No synthetic penalty objective is supplied. The next proposal uses the
  existing parent distributions and a new random draw.
- **Reproducibility:** each optimizer has its own RNG state. DEAP's global NumPy
  RNG is temporarily swapped and restored during generation; this adapter is
  intended for the existing serial runner, not concurrent threads.

`details.json` records the parent index, parent count, step size, completed
updates, and number of reflected coordinates for each CMA proposal. The manifest
records the DEAP version and the CMA protocol. The reporting Pareto archive
contains all successful evaluations, independently of the retained parents.

## Run on the actual AEMEC/OpenFOAM model

In the configured Linux/OpenFOAM environment, install benchmark dependencies:

```bash
python -m pip install -r opt/requirements-benchmark.txt
```

The benchmark uses the existing `AemecOpenFoamEvaluator`: each attempt performs a
fresh-case voltage sweep and obtains voltage/crossover at 10,000 A/m². The source
case is preserved, and scratch/log directories are isolated by benchmark, seed,
and algorithm. Normalization endpoints and variable bounds come from the supplied
analysis metadata; the original evaluated rows are **not** given to the optimizers.

First inspect the planned evaluation count without running CFD:

```bash
python opt/run_benchmark.py --backend aemec \
  --analysis ELA/results/analysis --case run/AEMEC \
  --output opt/results/benchmark_aemec \
  --budget 64 --initial 16 --seeds 42 43 44 45 46 --dry-run
```

Remove `--dry-run` to execute. With these settings the actual number of evaluations
is `5 × (16 + 4 × (64 - 16)) = 1040`, because initial observations are shared.
Each algorithm is charged for all 64 attempts. Start with a smaller budget/seed
count when checking the CFD setup. `--timeout-minutes`, `--mesh-command`,
`--solver-command`, `--work-dir`, `--ptl-side`, and the existing sweep stability
tolerances are configurable. Use settings consistent with the source case and
the analysis from which the bounds/scales were obtained.

## Comparison protocol

- **Shared initial data:** a scrambled Sobol design per seed, evaluated once and
  supplied identically to every algorithm. Prefer a power of two for `--initial`.
  GP BO and TPE start adaptive suggestions after this phase. The same observations
  form NSGA-II's first population (failed trials can delay completion of it), and
  successful ones form MO-CMA-ES's parent population.
- **Equal total attempt budget:** initialization, failed evaluations, and duplicate
  proposals all count. No algorithm receives free replacement attempts. Failed
  observations are marked failed and omitted from models/fronts; no arbitrary
  penalty responses are invented. Subsequent observations are not shared between
  algorithms. Duplicate adaptive proposals are evaluated again and charged.
- **Paired seeds:** use the same seed list for all algorithms. Summaries give
  descriptive statistics and paired differences; they do not claim statistical
  significance. More seeds and evaluation budgets are needed for reliable rankings.
- **Sequential execution:** every algorithm runs serially. NSGA-II retains its
  generation-based population selection; a partial last generation stops at the
  same attempt budget, without hidden evaluations past the cutoff. MO-CMA-ES
  updates after each successful offspring and needs no batch completion.
- **No optimizer sees future objective values:** models use only observations
  available in their own history. The predeclared bounds, normalization and
  reference are common study settings.

The GP acquisition scores a new scrambled Sobol candidate pool per step plus
corners and center, excluding previous attempted designs. It uses a plug-in
observed front and independent latent-function posterior variances, subtracting
the GP's fitted observation noise. This is q=1 EHVI, not noise-integrated NEHVI.
It has no global acquisition-maximization guarantee. `--gp-candidates` and
`--gp-restarts` tune numerical search effort; runtime is recorded separately.
If fewer than four successful observations exist, GP BO uses an explicitly logged
Sobol fallback. If every acquisition value is zero it selects maximum predictive
variance. GP hyperparameter-fit warnings and kernels are saved in `details.json`.

## Hypervolume and objective scaling

Both objectives are minimized. The primary metric is hypervolume with **one fixed
reference point and one fixed affine normalization for the whole benchmark**.
Larger values are better. No run uses its own final front to redefine the scale.

Defaults:

- Proxies: normalization endpoints [0,0] and [1,1], raw reference [1.1,1.1].
- AEMEC: the original ELA objective normalization endpoints; reference = lower +
  1.1 × (upper − lower), in volts and mol/s.
- ZDT1: endpoints [0,0] and [1,10], raw reference [1.1,11].

Use `--objective-lower LOW1 LOW2 --objective-upper HIGH1 HIGH2` to override the
fixed scaling, and `--reference REF1 REF2` for a fixed raw-unit reference. Set
these before comparing algorithms and keep them identical across comparison
runs. The reference should be worse than the region of practical interest.
Points outside its dominated rectangle contribute no hypervolume and are counted
in `outside_reference`. Negative normalized values and hypervolumes above one
are possible; values are not clipped.

`mean_hypervolume_over_budget` is trapezoidal area under the hypervolume curve
(starting with zero at attempt 0), divided by the total attempt budget. It rewards
early progress. It is not normalized AOCC against a known optimum. No true Pareto
front or IGD reference set is assumed.

## Outputs

- `algorithm_summary.csv`: mean, standard deviation, median, min/max across seeds,
  including final hypervolume, mean hypervolume over budget, successes/failures,
  Pareto size and timings. Standard deviation is blank for a single seed.
- `run_summary.csv`: one row per algorithm/seed.
- `paired_hypervolume.csv`: per-seed hypervolumes and paired differences.
- `history.csv`, `hypervolume.png`: convergence; plot shows means and seed IQRs.
- `all_run_pareto.csv`, `pareto_fronts.png`: final per-run nondominated points.
- `seed_*/shared_initial/`: shared initial designs, responses, failures, metadata.
- `seed_*/{gp,tpe,nsga2,cmaes}/`: complete evaluation trace, history, Pareto set,
  acquisition/model diagnostics, and CFD logs when applicable.
- `manifest.json`: backend/artifact hashes, versions, settings, normalization,
  reference, run status, actual evaluator calls and whole-benchmark wall time.

Evaluation and proposal/update times are reported separately. Shared initial
evaluation time is charged in full to each algorithm; `accounted_total_s` is that
charged evaluation time plus optimizer time, excluding CSV/plotting/I/O. It is not
the measured whole-process wall time. The benchmark physically saves initial
evaluations without making any algorithm's charged budget smaller.

Existing output directories are never overwritten. Completed evaluations are
checkpointed each step. Interruption preserves checkpoints, but exact sampler/RNG
resume is not implemented; start a new directory to repeat a benchmark.

## References and verification

- [Optuna TPESampler (v4.9)](https://optuna.readthedocs.io/en/v4.9.0/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [Optuna NSGAIISampler (v4.9)](https://optuna.readthedocs.io/en/v4.9.0/reference/samplers/generated/optuna.samplers.NSGAIISampler.html)
- [Optuna SBXCrossover (v4.9)](https://optuna.readthedocs.io/en/v4.9.0/reference/samplers/generated/optuna.samplers.nsgaii.SBXCrossover.html)
- [DEAP multiobjective CMA-ES strategy](https://deap.readthedocs.io/en/master/api/algo.html#deap.cma.StrategyMultiObjective)

Run the benchmark tests from the repository root:

```bash
PYTHONPATH=opt:. MPLCONFIGDIR=/tmp/aemec-matplotlib \
  ELA/.venv39/bin/python -m unittest opt.tests.test_benchmark opt.tests.test_benchmark_cmaes -v
```

Tests check deterministic and Monte Carlo EHVI agreement, Pareto/hypervolume
geometry, shared initialization, NSGA-II generation advancement, failed-attempt
accounting, reproducibility, output preservation and the mocked CFD adapter.
CMA tests additionally cover covariance adaptation, fixed-reference selection,
bound reflection, RNG isolation, recovery after failed initialization, missing
dependency checks, and exact execution of an odd attempt budget.
