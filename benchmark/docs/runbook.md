# Benchmark GP BO, TPE BO, NSGA-II, and CMA-ES

`benchmark/run_benchmark.py` compares five available algorithms on the same three-variable,
two-objective minimization problem. It leaves `opt/run_optimization.py` and existing
studies unchanged. Inputs remain membrane thickness, water inlet temperature,
and anode PTL porosity by default.

**CFD is the only benchmark evaluator.** Every shared initial design and every
subsequent proposal is evaluated with the AEMEC solver. Algorithm models may
score many cheap candidates internally, but only CFD responses enter evaluation
tables, Pareto sets, and hypervolume. `--backend aemec` is optional compatibility
syntax; `--backend proxies` and `--backend zdt1` are rejected.

| Algorithm key | Implementation |
| --- | --- |
| `gp` | Two independent Matérn-5/2 Gaussian processes with ARD; analytic expected hypervolume improvement (EHVI) |
| `tpe` | Optuna multivariate multiobjective TPE (MOTPE), 64 EI candidates |
| `nsga2` | Optuna NSGA-II, simulated binary crossover (eta=20, probability 0.9) and per-variable random-reset mutation (probability 1/3) |
| `cmaes` | DEAP multiobjective CMA-ES (MO-CMA-ES), one offspring per update, Pareto-rank and fixed-reference hypervolume selection |
| `ela_proxy` | Direct search on frozen ELA proxies; randomly weighted augmented Tchebycheff selection; every proposal assessed by CFD |

NSGA-II is the standard name for the requested “NSAG-II”. Its population size
equals `--initial`. This is Optuna's NSGA-II variant; its mutation resets a
selected variable, rather than using polynomial mutation.
The benchmark requires Optuna 4.9; queued initial observations are explicitly
assigned to population 0 using its generation API, so they participate in evolution.

The four baselines run by default. Add `ela_proxy` explicitly with its saved models. Select a subset with `--algorithms`; for
example, `--algorithms gp tpe nsga2` reproduces the previous three-method setup.

## Run on the actual AEMEC/OpenFOAM model

On your university server, from an activated Python environment:

```bash
cd "$HOME/AEMEC"
module load mpi/openmpi-x86_64
source "$HOME/OpenFOAM/OpenFOAM-v2306/etc/bashrc"
python -m pip install -r benchmark/requirements.txt
command -v blockMesh openFuelCell
```

The custom AEMEC solver and tools must also have been built with `./Allwmake -j 2`
from `src/`. Base OpenFOAM alone does not provide `openFuelCell` or the AEMEC
libraries. Before evaluation, the benchmark checks the mesh/solver executables,
including commands used inside the default `make mesh` recipe. This catches
missing commands but does not prove that linked libraries and case settings work.

Start with a small **actual CFD pilot** for the four baseline algorithms:

```bash
python benchmark/run_benchmark.py --backend aemec --case run/AEMEC \
  --algorithms gp tpe nsga2 cmaes \
  --budget 8 --initial 4 --seeds 42 --cma-sigma 0.2 \
  --output benchmark/results/benchmark_aemec_pilot
```

This makes **20 design-evaluation attempts**: four shared initial designs and
four further designs per algorithm. Inspect successes/failures in
`run_summary.csv` and the CFD logs before a full study. A failed attempt still
consumes budget. Use a fresh output directory for every run.

The full comparison requested earlier is:

```bash
python benchmark/run_benchmark.py --backend aemec --case run/AEMEC \
  --algorithms gp tpe nsga2 cmaes \
  --budget 64 --initial 16 --seeds 42 43 44 45 46 --cma-sigma 0.2 \
  --output benchmark/results/benchmark_aemec
```

This makes **1040 design-evaluation attempts**, computed as
`5 × (16 + 4 × (64 - 16))`. Each algorithm is charged all 64 attempts per seed,
while the initial observations are evaluated once per seed and shared.

Each attempt uses `AemecOpenFoamEvaluator` to copy a clean scratch case, update
the three design variables, generate the mesh, and run a **complete voltage
sweep**. Voltage and crossover at 10,000 A/m² are interpolated from stable sweep
holds and stored in V and mol/s. These are CFD model experiments, not laboratory
measurements. The source case is preserved; scratch cases and logs are isolated
by benchmark, seed, and algorithm.

Add `--dry-run` to either command to inspect the configuration and planned count
without running any evaluation. A dry run deliberately skips executable/runtime
checks so it also works on a machine without OpenFOAM.

### Portable study settings; no ELA results required

The default bounds and fixed objective scaling are shipped in
[`benchmark/configs/aemec_benchmark.json`](../configs/aemec_benchmark.json):

| Input | Lower | Upper |
| --- | ---: | ---: |
| Membrane thickness (µm) | 10 | 100 |
| Water inlet temperature (K) | 298.15 | 353.15 |
| Anode PTL porosity | 0.4 | 0.8 |

The objective scaling endpoints were copied from the existing ELA analysis;
they are historical normalization settings, not surrogate predictions or
guaranteed global objective limits. The original `ELA/results` files are **not
needed**. No previous evaluated rows are passed to any optimizer. The manifest
records the chosen configuration path/hash, actual bounds/scales, source-case
hash, and solver settings.

To change the study, copy/edit the JSON and pass `--problem-config FILE`.
Alternatively, `--analysis DIRECTORY` loads that directory's `metadata.json`
for compatibility with previous runs. These options are mutually exclusive.
`objective_limits` contains two rows: `[voltage_lower, crossover_lower]` and
`[voltage_upper, crossover_upper]`. Select scaling/reference settings before
comparing algorithms and keep them fixed throughout the comparison.

`--timeout-minutes`, `--mesh-command`, `--solver-command`, `--work-dir`,
`--ptl-side`, target current, and sweep stability tolerances remain configurable.
Custom mesh commands are checked by their entry executable; the caller must
provide their internal dependencies. Keep the case, objective definition, and
study settings consistent across algorithms.

## ELA proxy proposal algorithm

`ela_proxy` is a fifth algorithm, parallel to GP BO, TPE, NSGA-II, and CMA-ES.
Its proxy functions guide the choice of x. The common CFD evaluator supplies
cell voltage and crossover for that x, exactly as for every other algorithm.

Run all five, from the repository root in your OpenFOAM/Python environment:

```bash
python benchmark/run_benchmark.py \
  --algorithms gp tpe nsga2 cmaes ela_proxy \
  --proxy-root ELA/results/proxies_verified --proxy-rank 1 \
  --budget 8 --initial 4 --seeds 42 \
  --output benchmark/results/aemec_five_pilot
```

This pilot needs **24 CFD attempts**. For the full study, use `--budget 64
--initial 16 --seeds 42 43 44 45 46` and a fresh output directory: **1280 CFD
attempts** across five algorithms. Historical data used to create the proxies
are additional prior information; they are not included in these online counts.

The selected directory must contain both
`cell_voltage_v/proxy_01/proxy.json` and
`crossover_rate_mol_s/proxy_01/proxy.json`. These files must exist on the machine
running the benchmark. Copy the generated artifacts to the server if necessary.
The JSON roles and input bounds must match the CFD problem. They do not supply
objective scaling or change the CFD evaluator.

`--proxy-files VOLTAGE_JSON CROSSOVER_JSON` selects explicit files instead of a
root. Without either path option, discovery accepts one complete same-rank pair
under `ELA/results`; multiple pairs require an explicit choice. Missing models
fail before any CFD attempt. Add `--dry-run` to validate paths/configuration.
Selecting only `--algorithms ela_proxy` still runs CFD for every evaluated design.

### Search rule and feedback

At each adaptive step:

1. Build a fresh seeded Sobol pool of `--proxy-candidates` designs (default 2048),
   augmented with the eight corners and the center. Exclude all attempted
   designs, including failed CFD attempts.
2. Score the two frozen proxy expressions at those designs. Nonfinite scores
   are excluded from ranking; these are internal model errors, not CFD attempts.
3. Normalize each proxy objective to its range in this candidate pool, handling
   constant objectives without division by zero. This normalization is **only
   for proposal selection**; the benchmark's CFD scaling/reference stay fixed.
4. Draw positive random weights w from a seeded Dirichlet(1,1) distribution and
   choose the candidate minimizing `max(w * f_proxy) + 0.05 * sum(w * f_proxy)`.
   This augmented Tchebycheff score explores different compromises. Break score
   ties by the largest distance to prior attempted designs.
5. Run CFD at the selected design. Record the CFD response or failure, update
   the measured Pareto set/hypervolume, and exclude the attempted design from
   future proposals. If all proxy candidates are invalid, use an explicitly
   logged space-filling fallback proposal; it is still evaluated by CFD.

This is a finite-candidate direct-search baseline, with no global optimum
claim. Proxies remain frozen: the method does not refit them using CFD feedback.
CFD feedback assesses performance and marks attempted points. ELA matching alone
does not ensure that proxy minimizers or joint trade-offs coincide with CFD;
that is why the benchmark evaluates every selected design on the physical model.
This implementation does not train or generate an optimizer with LLaMEA.

Cheap proxy scores, weights, candidate counts, and cumulative mathematical
function-call counts appear only in `details.json`. Their computation time is
included in optimizer time. Raw observations and all comparison metrics contain
CFD results only. Failed CFD calls are charged and have no replacement values.
The manifest hashes the model files and records source manifests and historical
sample counts when available; missing provenance means unknown prior cost.
Comparisons share the online CFD budget, but are not equal in prior information
or total historical data acquisition cost.

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

All algorithms, including `ela_proxy`, use fixed endpoints from the bundled
problem configuration (or an explicit configuration/analysis). The default
reference is lower + 1.1 × (upper − lower), in volts and mol/s. The bundled
values preserve the original ELA normalization.

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
- `seed_*/{gp,tpe,nsga2,cmaes,ela_proxy}/`: complete evaluation trace, history, Pareto set,
  acquisition/model diagnostics, and CFD logs when applicable.
- `manifest.json`: CFD configuration and algorithm-asset hashes, versions, settings, normalization,
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
PYTHONPATH=opt:benchmark:. MPLCONFIGDIR=/tmp/aemec-matplotlib \
  python -m unittest discover -s benchmark/tests -v
```

Tests check deterministic and Monte Carlo EHVI agreement, Pareto/hypervolume
geometry, shared initialization, NSGA-II generation advancement, failed-attempt
accounting, reproducibility, output preservation and the mocked CFD adapter.
CMA tests additionally cover covariance adaptation, fixed-reference selection,
bound reflection, RNG isolation, recovery after failed initialization, missing
dependency checks, and exact execution of an odd attempt budget.
Path tests cover relocated/nested proxy folders, requested ranks, ambiguous or
incomplete pairs, explicit JSON paths, and discovery during a dry run.
AEMEC tests cover running without ELA result files, configuration overrides,
missing CFD executables, no fallback to proxies, and all four algorithms calling
the CFD adapter with the shared-initialization budget. The CFD adapter is mocked
in those tests; an actual solver run must be verified in the OpenFOAM environment.

Proxy-search tests additionally verify model-driven proposals, reproducibility,
failed-design exclusion, singular/constant model handling, and all five
algorithms recording only CFD responses and CFD-based metrics.
