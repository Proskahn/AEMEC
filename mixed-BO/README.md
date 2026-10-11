# Descriptor-informed mixed-variable Bayesian optimization

This is a separate GP-based Bayesian optimization workflow. The existing `opt/`
optimizer uses Optuna/TPE; it is unchanged. This folder keeps the three continuous
inputs and adds two categorical inputs, with three types each:

| Input column | Domain |
| --- | --- |
| `membrane_thickness_um` | 10–100 µm |
| `water_inlet_temperature_k` | 298.15–353.15 K |
| `ptl_porosity` | 0.40–0.80 |
| `anode_pte` | `AnodePTE_1`, `AnodePTE_2`, `AnodePTE_3` |
| `membrane_type` | `Membrane Type_1`, `Membrane Type_2`, `Membrane Type_3` |

The spelling **AnodePTE** follows the request. These names are identifiers; they
do not imply a particular material. Both objectives are minimized:
`cell_voltage_v` [V] and `crossover_rate_mol_s` [mol/s]. The search evaluates
all **nine valid material pairs** and proposes one new mixed design at a time.

## Representation: identity → latent → (identity, geometry, physics)

There is one small VAE for each categorical variable. For a material identity c:

```text
one_hot(c) → encoder → μ(c), log σ²(c)
                         ↓ reparameterized sample z during training
                      decoder
                   ↙     ↓      ↘
            category    ĝ(c)     p̂(c)
```

The three decoder targets are the original category, human-defined geometric
descriptors `g(c)`, and human-defined physical descriptors `p(c)`. The encoder
receives only the category identity, as requested. The augmented tuple is the
**decoder target**, not a requirement that the latent coordinates literally equal
the descriptor values. Two latent coordinates per categorical variable are the
default; `--latent-dim` changes that size.

The training loss is:

```text
cross_entropy(category)
  + geometry_weight × MSE(standardized geometry)
  + physics_weight × MSE(standardized physics)
  + beta × KL(q(z|c) || N(0,I))
```

Descriptor columns are standardized across the three catalog types. Identity
reconstruction has weight 1; geometry and physics weights default to 1 each.
The KL weight defaults to 0.01 and warms up over the first quarter of training.
No voltage/crossover values enter VAE training. After training, the representation
is frozen and each category is mapped to its **posterior mean**. GP inputs are:

```text
[thickness, water temperature, porosity, μ_anode(c_a), μ_membrane(c_m)]
```

Continuous inputs are mapped to [0,1] using the fixed study bounds. Each latent
family is centered and scaled by one scalar calculated from its three catalog
means, preserving relative Euclidean distances within that family. Two separate
Matérn 5/2 GPs, with learned length scales, predict the objectives. Objective
normalization and GP kernel fitting use training data only in each CV fold.
The known descriptor catalog can be used before CV because it contains no
held-out objective measurements. Do not derive descriptors from those objectives.

**With only three types per family, good reconstruction can be memorization.**
It does not establish meaningful material similarity, and reconstruction loss
alone does not guarantee that latent distances match descriptor distances.
Inspect the saved distance/reconstruction diagnostics and compare the supplied
baselines. GP uncertainty is conditional on this fixed embedding; it does not
propagate VAE posterior uncertainty. The encoder supports these known catalog
types; adding a new type requires updating the catalog and retraining.

## Installation and a runnable demonstration

From the repository root (Mac or Docker), use a separate environment:

```bash
python3 -m venv mixed-BO/.venv
mixed-BO/.venv/bin/python -m pip install -r mixed-BO/requirements.txt

mixed-BO/.venv/bin/python mixed-BO/run_mixed_bo.py demo \
  --output mixed-BO/results/my_demo --seed 42
```

On Linux, a CPU-only PyTorch wheel can be installed before the requirements:

```bash
mixed-BO/.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The demo trains the two VAEs, builds 36 **synthetic** mixed evaluations, trains
and cross-validates the two GPs, and proposes a new design. Its descriptor values
and objective functions are artificial software examples, not material data or
CFD results. Files are explicitly marked as demonstrations. Other commands
require `--allow-demo` to use these synthetic models/catalogs.

Output directories must be new; CSV outputs are never overwritten. Installation
is isolated from `ELA`, `GP`, and your existing optimization environment.

## Define the human descriptors

Copy `config/materials.template.json` to `config/materials.json`. For **each of
the six material types**, replace the `null` values, descriptor names, and units.
The template intentionally has no invented material properties. Extend each
family's schema and numeric arrays together. For example:

```json
"geometry": [{"name": "YOUR_DESCRIPTOR", "unit": "YOUR_UNIT"}],
"physics": [{"name": "YOUR_PROPERTY", "unit": "YOUR_UNIT"}],
"descriptors": {
  "AnodePTE_1": {"geometry": [null], "physics": [null]},
  "AnodePTE_2": {"geometry": [null], "physics": [null]},
  "AnodePTE_3": {"geometry": [null], "physics": [null]}
}
```

All types of one categorical variable must use the same descriptor definitions,
order, units, and reference conditions. Anode and membrane variables may have
different descriptors and different descriptor counts. Geometry and physics
each require at least one finite numeric descriptor. Descriptor values are
fixed material attributes at declared reference conditions. This implementation
does not model a descriptor as a function of temperature or variable porosity;
those remain separate GP inputs.

Train the descriptor representation **before fitting the GPs**:

```bash
mixed-BO/.venv/bin/python mixed-BO/run_mixed_bo.py fit-vae \
  --catalog mixed-BO/config/materials.json \
  --output mixed-BO/results/vae --epochs 2000 --latent-dim 2 --seed 42
```

`--geometry-weight`, `--physics-weight`, `--beta`, `--hidden-dim`, and
`--learning-rate` expose the model/loss choices. More training epochs do not add
material information when the catalog still has only three types per family.

The result includes `representation.json`, two `*_vae.pt` state dictionaries,
`embeddings.csv`, `descriptor_reconstruction.csv`, and `vae_training.csv`.
The JSON contains the catalog, normalization settings, latent posterior means
and variances, reconstruction errors, and all three pairwise distances per family.

## Obtain category-labeled evaluations

Generate initial proposals covering all nine type pairs:

```bash
mixed-BO/.venv/bin/python mixed-BO/run_mixed_bo.py sample \
  --catalog mixed-BO/config/materials.json --per-combination 4 \
  --output mixed-BO/results/initial_design.csv --seed 42
```

The output has 36 rows with blank objective values and `status=pending`.
Evaluate the correct materials and continuous settings, populate the objective
columns with actual results, and set successful rows to `status=complete`.
Save this evaluated table as `mixed-BO/results/evaluated.csv`.

**Existing ELA samples do not contain these two material identities.** You may
reuse them only after labeling the materials actually used. Do not randomly
assign types or copy the same CFD result onto different material combinations.
Data from one material pair alone cannot validate effects of the other types.
The GP report lists category coverage and warns when types have no observations.

This implementation exchanges proposals/evaluations through CSV files. It does
**not** map material labels to OpenFOAM files or launch CFD: the physical meaning
of the three material types and their solver configuration have not been supplied.
Descriptors influence the surrogate; they do not automatically configure the
physical simulation. Material-to-case mappings need to be defined before any
automated CFD loop can be added.

## Train, validate, and compare representations

```bash
mixed-BO/.venv/bin/python mixed-BO/run_mixed_bo.py train \
  mixed-BO/results/evaluated.csv \
  --representation mixed-BO/results/vae \
  --output mixed-BO/results/gp --folds 5 --seed 42
```

There must be at least 12 valid observations. Failed/pending or invalid-objective
rows are excluded with reasons; invalid category labels and duplicate complete
mixed designs are rejected. Continuous bounds remain the original study bounds.
After CV, final models use all accepted observations. Results include two joblib
models, `cv_metrics.csv`, `cv_fold_metrics.csv`, `cv_predictions.csv`,
`training_data.csv`, `excluded.csv`, `metadata.json`, and `report.md`.
All errors/predictions are in physical units. Only load trusted model artifacts.

Use the same observations and CV seed to compare:

| `--encoding` | Categorical GP inputs |
| --- | --- |
| `vae` (default) | Learned latent means |
| `descriptors` | Standardized human g/p values directly |
| `onehot` | Category identity only, with no material descriptors |

The category-only baseline uses a Matérn kernel on one-hot vectors; it is not
BoTorch's exact sum/product mixed Hamming kernel. No category is treated as an
ordered numeric material value. Choose a separate output directory per encoding.
Hyperparameter/encoding choices based on CV scores need independent validation
for an unbiased final accuracy claim.

`--cv-mode random` is the default. For transfer tests, use `--cv-mode anode_pte`,
`--cv-mode membrane_type`, or `--cv-mode combination`: these leave one observed
type or pair out at a time. In these modes the group count determines the number
of folds and `--folds` is ignored. Descriptor availability for held-out known
types is assumed; their objective observations are excluded from that fold.

## Propose the next evaluation and continue

```bash
mixed-BO/.venv/bin/python mixed-BO/run_mixed_bo.py suggest \
  --model-dir mixed-BO/results/gp \
  --output mixed-BO/results/proposal_01 --seed 42
```

The acquisition is analytic **two-objective expected hypervolume improvement
(EHVI)** for independent GP objectives, both minimized. It uses latent-function
posterior variance (fitted observation-noise variance removed) and the observed
Pareto front. This is a plug-in, single-point acquisition, not noise-integrated
NEHVI or joint batch EHVI. The default reference is the worst observed value plus
10% of the observed range per objective. Override with
`--reference-point VOLTAGE_V CROSSOVER_MOL_S` to fix it across iterations.

For each of the nine valid pairs, the search scores a finite scrambled Sobol
pool plus corners and the center of the continuous domain. Increase
`--candidates-per-combination` (default 512) for a denser search; this is a
candidate-pool search, not a guaranteed global acquisition maximum. Candidates
never contain fractional/fictional categories or arbitrary decoded materials.
Already evaluated designs are excluded. `--pending designs.csv` excludes designs
already proposed; it does not model pending outcomes with fantasies.

`next_design.csv` contains the single selected design, predictions, blank actual
objectives, and `status=pending`. `candidate_scores.csv` contains all ranked
alternatives; it is **not** an independently optimized batch. Evaluate the
selected design and save its real results in `new_evaluation.csv`, then append:

```bash
mixed-BO/.venv/bin/python mixed-BO/run_mixed_bo.py append \
  mixed-BO/results/evaluated.csv \
  --new-evaluations mixed-BO/results/new_evaluation.csv \
  --catalog mixed-BO/config/materials.json \
  --output mixed-BO/results/evaluated_next.csv
```

Run `train` on the updated CSV with the same frozen representation and a new GP
output directory, then `suggest` again. If material descriptors change, retrain
the VAEs and GPs together. Saved hashes reject a changed representation paired
with older GP models. `predict designs.csv --model-dir ... --output ...` also
provides standalone predictions for valid known materials within the study bounds.

## Verification and references

```bash
mixed-BO/.venv/bin/python -m unittest discover -s mixed-BO/tests -v
```

Tests check descriptor gradients and reconstruction, changed-descriptor effects,
CV target isolation, category-held-out splits, valid proposals, model persistence,
and EHVI against deterministic hypervolume differences and Monte Carlo integration.

Method references: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114),
[BoTorch mixed GP kernel documentation](https://botorch.readthedocs.io/en/stable/_modules/botorch/models/gp_regression_mixed.html),
and [EHVI box-decomposition reference implementation](https://botorch.readthedocs.io/en/stable/_modules/botorch/acquisition/multi_objective/analytic.html).
This implementation uses PyTorch for the VAE and scikit-learn GPs with its own
two-dimensional EHVI integration; it does not depend on BoTorch.
