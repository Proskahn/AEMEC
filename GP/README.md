# Gaussian-process surrogates for AEMEC

Two independent Gaussian processes map the same three design variables to cell
voltage and gas crossover, using **256 evaluated samples** from ELA.

| Input column | Unit |
| --- | --- |
| `membrane_thickness_um` | µm |
| `water_inlet_temperature_k` | K |
| `ptl_porosity` | Fraction, anode PTL |

| Output column | Unit |
| --- | --- |
| `cell_voltage_v` | V |
| `crossover_rate_mol_s` | mol/s |

Keep the operating conditions consistent across samples. The ELA evaluator uses
anode PTL porosity and 10000 A/m² by default. Crossover is the evaluator's H₂
molar rate, not a gas mole fraction.

## Run training and cross-validation

From the repository root, in a Python 3.9–3.11 environment:

```bash
python -m pip install -r GP/requirements.txt
python GP/train_gp.py ELA/results/evaluated_256.csv \
  --output GP/results --folds 5 --seed 42
```

Both paths shown above are defaults, so `python GP/train_gp.py` also works.
The CSV must contain the three inputs and two **raw, physical-unit** outputs.
Use evaluated samples or ELA's `accepted.csv`, not `design.csv` or `features.csv`.
If all 256 evaluations ran as one batch, pass
`ELA/results/evaluation/evaluated.csv` instead.

For the existing 128 + 128 evaluation batches, combine them once:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

destination = Path('ELA/results/evaluated_256.csv')
if destination.exists():
    raise SystemExit(f'{destination} already exists; use the existing file')
frames = []
for batch in ['evaluation', 'evaluation_additional']:
    frame = pd.read_csv(Path('ELA/results') / batch / 'evaluated.csv')
    frame['source_batch'] = batch
    frames.append(frame)
pd.concat(frames, ignore_index=True).sort_values('sample_id').to_csv(destination, index=False)
PY
```

Training requires exactly 256 successful, finite, physically valid rows and
rejects duplicate designs, duplicate IDs, mixed operating settings, and
degenerate input/output data. Failed or pending evaluations are never converted
to penalty values. If exclusions leave fewer than 256 rows, finish those
evaluations first. `--expected-samples N` explicitly permits another study size.
Existing nonempty result directories are preserved; use a different `--output`
for another run. No CFD evaluations are launched by the GP scripts.

## Models and validation

Each output has its own GP with a constant-amplitude **Matérn 5/2 kernel**, a
separate learned length scale for each input, and a small learned white-noise
term. Kernel hyperparameters are fitted by maximizing log marginal likelihood,
with three random optimization restarts plus the initial optimization. Set
`--restarts` to change that computational budget.

Inputs are standardized to zero mean and unit variance. Each objective is also
standardized independently with `normalize_y=True`. In cross-validation,
**all scaling and kernel fitting use only the training fold**. Returned means,
standard deviations, errors, and intervals are converted back to physical units.
Do not first apply ELA's whole-dataset objective normalization.

Five-fold shuffled CV uses the same splits for both objectives: each sample
receives one prediction from a GP trained without that sample. Each training
fold has 204 or 205 samples; its held-out fold has 52 or 51. Aggregate metrics use
the pooled held-out predictions. A linear regression baseline uses the same
folds, so you can check whether the GP improves on the nearly linear landscape.
After validation, the two final models are fitted on all 256 samples and saved.

CV estimates interpolation performance under this sampled design distribution.
It does not establish accuracy outside the sampled domain. Choosing kernels or
other modeling options based on these scores would require further independent
or nested validation for an unbiased comparison.

| Result | Contents |
| --- | --- |
| `cell_voltage_v.joblib` | Final voltage GP with input scaler |
| `crossover_rate_mol_s.joblib` | Final crossover GP with input scaler |
| `cv_metrics.csv` | Pooled held-out R², RMSE, MAE, maximum error, RMSE/range, and GP interval coverage |
| `cv_fold_metrics.csv` | The same metrics separately for each fold |
| `cv_predictions.csv` | Observations, fold assignments, GP predictions/intervals, and linear predictions |
| `cross_validation.png`, `.pdf` | Held-out prediction versus observation and residual plots |
| `training_data.csv`, `excluded.csv` | Accepted samples and exclusion audit |
| `metadata.json` | Source hash, versions, seed, fold indices, fitted kernels/scales, and optimization warnings |
| `report.md` | Validation summary and interpretation limits |

Nominal 95% predictive intervals are mean ± 1.96 standard deviations, including
the fitted white-noise term. They condition on the optimized kernel and do not
represent uncertainty in the underlying physical model. Inspect their held-out
coverage: narrow intervals need not be well calibrated. Predictions and interval
bounds for crossover are not clipped at zero. Kernel-bound and convergence
warnings are recorded in `metadata.json` and counted in the report.

## Predict new designs

```bash
python GP/predict_gp.py ELA/results/design_new.csv \
  --model-dir GP/results --output GP/predictions.csv
```

The design CSV needs the three input columns. The output preserves its columns
and adds a mean, standard deviation, and nominal 95% limits for each objective.
The script rejects inputs beyond the observed minimum/maximum of any input
unless you pass `--allow-extrapolation`; such predictions are flagged. Being
inside these marginal ranges does not guarantee adequate nearby sample coverage.
Only load trusted joblib models and use the dependency versions recorded during
training.

## Tests

```bash
python -m unittest discover -s GP/tests -v
```

Tests use synthetic functions to check fold isolation, physical output units,
input validation, model persistence, and the complete training/prediction workflow.
