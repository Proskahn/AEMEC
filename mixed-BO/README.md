# Physics-shaped full-design VAE + mixed-variable Bayesian optimization

The default representation now implements a **single full-design VAE** with
auxiliary geometric and physics supervision. Both objectives are minimized:
cell voltage `cell_voltage_v` [V] and crossover `crossover_rate_mol_s` [mol/s].

| Design input | Domain |
| --- | --- |
| `membrane_thickness_um` | 10–100 µm |
| `water_inlet_temperature_k` | 298.15–353.15 K |
| `ptl_porosity` | 0.40–0.80 |
| `anode_pte` | `AnodePTE_1`, `AnodePTE_2`, `AnodePTE_3` |
| `membrane_type` | `Membrane Type_1`, `Membrane Type_2`, `Membrane Type_3` |

## Model and exact loss

For mixed inputs, the numeric design vector is

```text
x_num = [three continuous values scaled to [0,1],
         one_hot(anode_pte), one_hot(membrane_type)]    # 9 components

x_num → encoder → μφ(x), ℓφ(x)
                      ↓ z = μφ(x) + exp(ℓφ(x)/2) ⊙ ε,   ε ~ N(0,I)
                      ├── design decoder → x̂_num
                      ├── geometry head → standardized ĝ(x)
                      └── physics head  → standardized p̂(x)
```

This defines `qφ(z|x) = N(μφ(x), diag(exp ℓφ(x)))`. The encoder receives only
`x_num`. The auxiliary quantities are **supervision targets**, not encoder inputs.
The decoder reconstructs the whole design, including its continuous values.
The categorical blocks are reconstructed with squared error against their one-hot
vectors; categorical cross entropy is not used. Argmax decoding is only for
reconstruction diagnostics, never for generating BO candidates.

```text
L_rec = batch_mean(sum_over_design_coordinates((x_num - x̂_num)²))
L_KL  = batch_mean(-0.5 × sum_over_latent_coordinates(1 + ℓ - μ² - exp(ℓ)))
L_g   = mean_squared_error(ĝ, standardized_g)
L_p   = mean_squared_error(p̂, standardized_p)

L_shape = L_rec + β L_KL + λ_g L_g + λ_p L_p
```

Defaults: latent dimension 4, hidden dimension 64, β = 0.01, λ_g = λ_p = 1,
Adam learning rate 0.001, minibatch size 128, and 2000 epochs. The loss weights
are fixed throughout training; there is no implicit KL warm-up. Nonnegative
weights, including zero for ablations, are supported. Log variance is clamped
between -10 and 6 for numerical stability. Auxiliary columns are standardized
using only the cheap pretraining dataset; constant columns use scale 1.

After pretraining, the model enters evaluation mode with gradients disabled.
The GP receives **exactly `T_shape(x) = μφ(x)`**: no random latent draws, no
appended raw continuous variables, and no extra latent rescaling. Thus a
four-dimensional latent model has four GP inputs. The two independent Matérn
GPs normalize their objective values using only GP training observations.

**No expensive y is used for representation learning.** Pretraining explicitly
selects only design and auxiliary columns. Even if voltage, crossover, status,
or predictions are present in the input file, they do not affect filtering,
standardization, training, or saved pretraining targets. Auxiliary definitions
must themselves be inexpensive and independent of high-fidelity responses.

## Run the synthetic demonstration

From the repository root, create an environment if one is not already installed:

```bash
python3 -m venv mixed-BO/.venv
mixed-BO/.venv/bin/python -m pip install -r mixed-BO/requirements.txt
```

For Linux CPU-only PyTorch, install it before the requirements with:

```bash
mixed-BO/.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Run the complete demonstration:

```bash
mixed-BO/.venv/bin/python mixed-BO/run_mixed_bo.py demo \
  --output mixed-BO/results/my_shape_demo \
  --latent-dim 4 --epochs 2000 --seed 42
```

The demo generates **576 cheap designs** with synthetic g(x)/p(x), pretrains the
VAE, then separately generates **36 synthetic expensive evaluations** for GP
training/CV and proposes the next design. `--pretrain-per-combination` controls
the cheap data count per material pair (default 64); `--per-combination` controls
the synthetic GP observation count per pair (default 4). Neither set is CFD data.
The cheap quantities are generated without calling the synthetic expensive
objective function. The two design pools use different seeds.

## Human-defined g(x) and p(x) for real use

`config/categories.json` defines the six category names and needs no invented
material properties. Supply a cheap pretraining CSV containing:

- The five original design columns listed above.
- One or more columns starting with `g__` for geometric descriptors.
- One or more columns starting with `p__` for inexpensive physics proxies.

Start from `config/pretraining.template.csv`, or generate unlabeled designs:

```bash
mixed-BO/.venv/bin/python mixed-BO/run_mixed_bo.py sample \
  --catalog mixed-BO/config/categories.json --per-combination 64 \
  --output mixed-BO/results/cheap_designs.csv --seed 42
```

Compute your own g(x)/p(x), add their columns, and save as
`mixed-BO/results/cheap_pretraining.csv`. These quantities may depend on the
**entire design**, including thickness, temperature, porosity, and both materials.
Use consistent definitions, units, and reference conditions for every row;
include units in column names or accompanying study documentation. Both
auxiliary families must be present and finite. At least 12 unique designs,
coverage of all categorical levels, and variation spanning all three continuous
inputs are required. Expensive evaluations and successful CFD status are not
required at this stage. The sample command's blank objective columns can be removed.

Pretrain the VAE:

```bash
mixed-BO/.venv/bin/python mixed-BO/run_mixed_bo.py fit-vae \
  --catalog mixed-BO/config/categories.json \
  --pretrain-csv mixed-BO/results/cheap_pretraining.csv \
  --output mixed-BO/results/shape_vae \
  --latent-dim 4 --epochs 2000 --batch-size 128 \
  --beta 0.01 --geometry-weight 1 --physics-weight 1 --seed 42
```

`--geometry-weight` and `--physics-weight` are λ_g and λ_p. Set both to zero
for a reconstruction/KL-only VAE ablation. `--hidden-dim`, `--learning-rate`,
and `--batch-size` are configurable independently.

For backwards-compatible **data preparation only**, you may still fill
`config/materials.template.json` with static material properties and omit
`--pretrain-csv`. The command will generate 64 full designs per material pair
and obtain auxiliary targets by catalog lookup. This also trains one full-design
VAE. Catalog-only auxiliaries depend on material identities and do not express
continuous geometry/physics dependence; per-design g__/p__ columns take precedence.

## Saved representation and GP training

The VAE bundle contains:

| File | Contents |
| --- | --- |
| `representation.json` | Architecture, frozen weights, design transform, auxiliary scalers, hashes, and diagnostics |
| `shape_vae.pt` | PyTorch state dictionary for the full model |
| `pretraining_data.csv` | Only the x/g/p values actually used |
| `vae_training.csv` | All four loss terms, total loss, and weights by epoch |
| `embeddings.csv` | Deterministic latent means and posterior standard deviations for cheap designs |
| `pretraining_reconstruction.csv` | Reconstructed design and auxiliary quantities at the posterior mean |

Reconstruction diagnostics describe the cheap training data, not held-out
high-fidelity accuracy. Review continuous reconstruction errors, categorical
accuracy, auxiliary errors, and latent variation for collapse or information loss.

Prepare a **separate expensive-evaluation CSV** with the five design columns,
actual `cell_voltage_v` and `crossover_rate_mol_s`, and optionally `status`.
Material labels must identify what was actually evaluated. Do not relabel or
copy existing ELA outputs onto other materials. Failed/pending observations are
excluded; duplicate mixed designs and invalid labels are rejected.

```bash
mixed-BO/.venv/bin/python mixed-BO/run_mixed_bo.py train \
  mixed-BO/results/evaluated.csv \
  --representation mixed-BO/results/shape_vae \
  --output mixed-BO/results/shape_gp --folds 5 --seed 42
```

At least 12 valid expensive observations are required. The encoder remains fixed
through every CV fold and the final fit. The result includes two GP models,
pooled/per-fold metrics, held-out predictions, accepted/excluded observations,
representation snapshot, and a report. GP output normalization and kernel fitting
use only each fold's training responses. GP uncertainty conditions on the fixed
representation; it does not propagate VAE posterior uncertainty.

`--cv-mode random` gives shuffled K-fold CV. `anode_pte`, `membrane_type`, and
`combination` leave one observed type or pair out at a time, ignoring `--folds`.
These are GP validation modes **conditional on pretraining**. If the cheap
pretraining pool contains held-out designs/types, their x/g/p are known
transductively, although their expensive responses are unused. Use a separate
pretraining pool when studying fully inductive generalization to new designs.

The `--encoding onehot` baseline gives the GP the original 9-component design.
`--encoding descriptors` remains a continuous-plus-static-material-descriptors
baseline and requires a populated material catalog; it does not evaluate general
per-design cheap functions at new candidates. Use separate output directories.
Compression and auxiliary supervision need not improve GP accuracy: good
reconstruction can still distort distances or discard information relevant to y.
Compare on appropriate held-out data; selecting representation settings from CV
scores requires further validation for an unbiased final performance claim.

## Propose, evaluate, and update

```bash
mixed-BO/.venv/bin/python mixed-BO/run_mixed_bo.py suggest \
  --model-dir mixed-BO/results/shape_gp \
  --output mixed-BO/results/proposal_01 --seed 42
```

The acquisition is single-point, two-objective analytic expected hypervolume
improvement (EHVI), with both objectives minimized. All nine valid category pairs
are enumerated; a finite Sobol pool plus corners/center covers continuous inputs.
Every candidate is encoded by the frozen full-design VAE. No acquisition search
or decoding occurs in arbitrary latent space. `--candidates-per-combination`
(default 512) controls pool density; there is no global-optimum guarantee.

The method uses independent latent-function GP posteriors with fitted observation
noise removed from their variance, and the observed Pareto front as a plug-in
baseline. It is not noise-integrated NEHVI or joint batch EHVI. The default
reference point is the worst observed value plus 10% of the observed range per
objective. Set `--reference-point VOLTAGE_V CROSSOVER_MOL_S` to fix it across runs.

`next_design.csv` contains one design, GP predictions, blank actual objectives,
and `status=pending`. `candidate_scores.csv` contains the alternatives.
Already evaluated points are excluded; `--pending designs.csv` also excludes
outstanding proposals, without modeling their unknown outcomes.

Evaluate the proposed design with the correct materials, fill its actual outputs,
set `status=complete`, and save as `new_evaluation.csv`. Then:

```bash
mixed-BO/.venv/bin/python mixed-BO/run_mixed_bo.py append \
  mixed-BO/results/evaluated.csv \
  --new-evaluations mixed-BO/results/new_evaluation.csv \
  --catalog mixed-BO/config/categories.json \
  --output mixed-BO/results/evaluated_next.csv
```

Retrain the GPs on the updated CSV with the same frozen representation and a new
output directory, then suggest again. `predict designs.csv --model-dir ...
--output ...` provides predictions without running the acquisition.

This remains a file-based BO workflow. Material-to-OpenFOAM mappings have not
been supplied, so no CFD is launched or material configuration inferred. For
synthetic catalogs/models, non-demo commands require `--allow-demo`. Output paths
must be new. Load only trusted joblib models and retain their recorded environment.

## Migration and verification

The representation schema is now **2**. Earlier per-category VAE/GP bundles are
preserved on disk but rejected by the updated loader with a retraining message.
Run `fit-vae` and `train` in new folders; old category-only models cannot be used
as full-design encoders. Changed representation weights/catalogs are also rejected.

```bash
mixed-BO/.venv/bin/python -m unittest discover -s mixed-BO/tests -v
```

Tests verify the requested loss numerically, no influence of y or status,
auxiliary supervision and zero-weight ablations, continuous sensitivity of the
latent representation, frozen posterior-mean-only GP inputs, serialization,
CV isolation, valid proposals, and EHVI against hypervolume/Monte Carlo checks.
The VAE uses [reparameterized variational inference](https://arxiv.org/abs/1312.6114);
EHVI is implemented by a two-objective non-dominated box decomposition, following
the method described in the [BoTorch analytic EHVI documentation](https://botorch.readthedocs.io/en/stable/_modules/botorch/acquisition/multi_objective/analytic.html).
The runtime uses PyTorch and scikit-learn; BoTorch is not a dependency.
