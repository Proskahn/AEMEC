# Benchmark GP BO, TPE BO, NSGA-II, and CMA-ES

`benchmark/run_benchmark.py` compares six available algorithms on the same three-variable,
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
| `cmaes` | Pinned OptunaHub MoCmaSampler, generation updates, native Pareto/hypervolume selection (see caveats below) |
| `ela_proxy` | Direct search on frozen ELA proxies; randomly weighted augmented Tchebycheff selection; every proposal assessed by CFD |
| `llamea` | LLaMEA evolves proposal code using candidate CFD hypervolume, then deploys the best within the same attempt budget |

NSGA-II is the standard name for the requested “NSAG-II”. Its population size
equals `--initial`. This is Optuna's NSGA-II variant; its mutation resets a
selected variable, rather than using polynomial mutation.
The benchmark pins Optuna 4.9.0; queued initial observations are explicitly
assigned to population 0 using its generation API, so they participate in evolution.

The four baselines run by default. Add `ela_proxy` explicitly with its saved models. Select a subset with `--algorithms`; for
example, `--algorithms gp tpe nsga2` reproduces the previous three-method setup.

LLaMEA is optional and requires Python 3.11+, additional requirements, and a chosen
LLM provider/model. Its candidate trials and winner deployment share the normal
CFD budget. See the [LLaMEA guide](llamea.md) for the protocol and run commands.

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
  --budget 8 --initial 4 --seeds 42 \
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
  --budget 64 --initial 16 --seeds 42 43 44 45 46 \
  --output benchmark/results/benchmark_aemec
```

This makes **1040 design-evaluation attempts**, computed as
`5 × (16 + 4 × (64 - 16))`. Each algorithm is charged all 64 attempts per seed,
while the initial observations are evaluated once per seed and shared.

Each attempt uses `AemecOpenFoamEvaluator` to copy a clean scratch case, update
the three design variables, generate the mesh, and run a **20-unit fixed-current
evaluation** at 10,000 A/m². Voltage and crossover are averaged over the final
5 solver-clock units after checking target current and stability of both
objectives, and stored in V and mol/s. Use `--simulation-duration-s 10` to
try a shorter run, and use a fresh output directory for each protocol.
`--evaluation-mode voltage-sweep` retains the old full-sweep/interpolation
method. See [settings and convergence checks](../../opt/docs/fixed-current.md).
These are CFD model experiments, not laboratory
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

Run these five methods, from the repository root in your OpenFOAM/Python environment:

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
`ela_proxy` searches the saved functions directly. The separate `llamea` method
generates optimizer code and is described in the [LLaMEA guide](llamea.md).

Cheap proxy scores, weights, candidate counts, and cumulative mathematical
function-call counts appear only in `details.json`. Their computation time is
included in optimizer time. Raw observations and all comparison metrics contain
CFD results only. Failed CFD calls are charged and have no replacement values.
The manifest hashes the model files and records source manifests and historical
sample counts when available; missing provenance means unknown prior cost.
Comparisons share the online CFD budget, but are not equal in prior information
or total historical data acquisition cost.

## How CMA-ES handles the two objectives

The `cmaes` option uses **OptunaHub's `MoCmaSampler`** through the same Optuna
ask/tell interface as TPE and NSGA-II. This is a contributed multiobjective
sampler, distinct from Optuna's single-objective `CmaEsSampler`. Both CFD
objectives remain separate; no weighted scalar objective is introduced.

The sampler source is bundled unchanged at commit
`7508d98688d3eb55aab09002ea6a8dedfbe5b2c6`, with its MIT license. Runs do not
fetch executable code from the network. Install `benchmark/requirements.txt`
to get the tested Optuna 4.9.0 runtime; neither DEAP nor OptunaHub is a runtime
dependency. The adapter checks compatibility before spending any CFD calls.

The implementation choices are:

- **Initialization:** `--initial` sets a fixed population size. Shared Sobol
  observations are enqueued and explicitly tagged by the sampler's generation
  bookkeeping. Only successful observations fill slots. If any initial calls
  fail, later proposals fill the missing slots with random points; these attempts
  consume the normal CFD budget. Even an all-failed initialization needs a full
  population of successful observations before CMA offspring generation begins.
- **Updates:** CFD runs serially, but the sampler selects parents and adapts
  after an entire generation has succeeded, at the next ask. A partial last
  generation stops at the exact budget without extra evaluations. Its successful
  responses still enter the reporting Pareto front and hypervolume.
- **Initial scale:** upstream fixes sigma at `1/6` of each normalized input range
  with identity covariance, then adapts per parent. Remove `--cma-sigma` from old
  commands: it now produces an explicit migration error rather than being ignored.
- **Bounds:** upstream clips Gaussian proposals to the unit box. Optuna may nudge
  the upper float endpoint inward by one representable step. These exact
  coordinates are evaluated, recorded and supplied back to the sampler.
- **Selection:** the sampler uses its native Pareto-rank/hypervolume code and
  a population-derived internal reference. The **reporting** reference remains
  fixed across every algorithm and seed; it is a separate benchmark setting.
- **Failures:** trials become Optuna `FAIL`, without penalty objectives. A failed
  slot remains unfilled until a later charged attempt succeeds. The population
  does not shrink and the budget is never extended to complete it.
- **Reproducibility:** each sampler owns its random generators; it does not swap
  global NumPy state. The static three-variable search space is declared up front.

`details.json` records generation, population size, instance slot and parent
trial ID for each proposal. `manifest.json` records the source commit, SHA-256,
Optuna version, protocol and caveats. Old DEAP results describe a different
implementation and should retain their original identity when comparing runs.

**Upstream caveats:** in this pinned source, the covariance update uses a scalar
vector dot product where an outer product is expected, and overflow-front
selection retains the smallest hypervolume contributor first. Both can affect
optimization quality. They are preserved to benchmark the requested upstream
implementation and are disclosed in every generated report/manifest. See the
[vendor notes](../aemec_benchmark/_vendor/mocma/README.md) for the exact operations.
The tests validate the integration and budget accounting; they do not certify
this sampler as a mathematically faithful reference implementation of MO-CMA-ES.

## Comparison protocol

- **Shared initial data:** a scrambled Sobol design per seed, evaluated once and
  supplied identically to every algorithm. Prefer a power of two for `--initial`.
  GP BO and TPE start adaptive suggestions after this phase. The same observations
  form NSGA-II's first population (failed trials can delay completion of it), and
  successful ones fill MO-CMA-ES's initial population; failures delay its completion.
- **Equal total attempt budget:** initialization, failed evaluations, and duplicate
  proposals all count. No algorithm receives free replacement attempts. Failed
  observations are marked failed and omitted from models/fronts; no arbitrary
  penalty responses are invented. Subsequent observations are not shared between
  algorithms. Duplicate adaptive proposals are evaluated again and charged.
- **Paired seeds:** use the same seed list for all algorithms. Summaries give
  descriptive statistics and paired differences; they do not claim statistical
  significance. More seeds and evaluation budgets are needed for reliable rankings.
- **Sequential execution:** every algorithm runs serially. NSGA-II and MoCmaSampler retain
  generation-based updates; a partial last generation stops at the same attempt
  budget, without hidden evaluations past the cutoff.
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
- [OptunaHub MoCmaSampler](https://hub.optuna.org/samplers/mocma/)
- [Pinned MoCmaSampler source](https://github.com/optuna/optunahub-registry/blob/7508d98688d3eb55aab09002ea6a8dedfbe5b2c6/package/samplers/mocma/mocma.py)

Run the benchmark tests from the repository root:

```bash
PYTHONPATH=opt:benchmark:. MPLCONFIGDIR=/tmp/aemec-matplotlib \
  python -m unittest discover -s benchmark/tests -v
```

Tests check deterministic and Monte Carlo EHVI agreement, Pareto/hypervolume
geometry, shared initialization, NSGA-II generation advancement, failed-attempt
accounting, reproducibility, output preservation and the mocked CFD adapter.
CMA tests additionally cover shared-trial population registration, generation
advancement/adaptation, clipping, RNG isolation, recovery after failed
initialization, pinned-source/runtime checks, the removed sigma option, and
exact execution of an odd attempt budget with the CFD adapter mocked.
Path tests cover relocated/nested proxy folders, requested ranks, ambiguous or
incomplete pairs, explicit JSON paths, and discovery during a dry run.
AEMEC tests cover running without ELA result files, configuration overrides,
missing CFD executables, no fallback to proxies, and all four algorithms calling
the CFD adapter with the shared-initialization budget. The CFD adapter is mocked
in those tests; an actual solver run must be verified in the OpenFOAM environment.

Proxy-search tests additionally verify model-driven proposals, reproducibility,
failed-design exclusion, singular/constant model handling, and all five
algorithms recording only CFD responses and CFD-based metrics.
