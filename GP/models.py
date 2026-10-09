"""Data validation, independent GP fits, and leakage-free cross-validation."""
from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

INPUTS = ['membrane_thickness_um', 'water_inlet_temperature_k', 'ptl_porosity']
OUTPUTS = ['cell_voltage_v', 'crossover_rate_mol_s']
UNITS = ['V', 'mol/s']


def load_data(path, expected_samples=256):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Evaluated data file not found: {path}. Copy the 256-point evaluated CSV from ELA here '
                         'or pass its actual path; design.csv and features.csv are not training data.')
    frame = pd.read_csv(path)
    missing = set(INPUTS+OUTPUTS)-set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns {sorted(missing)}; supply evaluated.csv, not design.csv or features.csv')
    if expected_samples < 16:
        raise ValueError('--expected-samples must be at least 16')
    values = frame[INPUTS+OUTPUTS].apply(pd.to_numeric, errors='coerce')
    reasons = pd.Series('', index=frame.index)
    if 'status' in frame:
        success = frame.status.astype(str).str.strip().str.lower().isin(['ok', 'success', 'complete', 'completed'])
        reasons.loc[~success] = 'unsuccessful_status'
    reasons.loc[(reasons == '') & ~np.isfinite(values.to_numpy()).all(axis=1)] = 'missing_or_nonfinite_value'
    physical = ((values[INPUTS[0]] <= 0) | (values[INPUTS[1]] <= 0) |
                (values[INPUTS[2]] <= 0) | (values[INPUTS[2]] >= 1) |
                (values[OUTPUTS[0]] <= 0) | (values[OUTPUTS[1]] < 0))
    reasons.loc[(reasons == '') & physical] = 'invalid_physical_value'
    excluded = frame.loc[reasons != ''].copy()
    excluded['exclusion_reason'] = reasons[reasons != '']
    accepted = frame.loc[reasons == ''].copy()
    accepted[INPUTS+OUTPUTS] = values.loc[accepted.index]
    if len(accepted) != expected_samples:
        raise ValueError(f'Expected {expected_samples} valid samples, found {len(accepted)} '
                         f'({len(excluded)} excluded). Finish/retry missing ELA evaluations, or explicitly '
                         'set --expected-samples for a different dataset size.')
    if accepted[INPUTS].duplicated().any():
        raise ValueError('Duplicate input designs would leak across CV folds; aggregate replicates explicitly first')
    if 'sample_id' in accepted and (accepted.sample_id.isna().any() or accepted.sample_id.duplicated().any()):
        raise ValueError('sample_id must be nonempty and unique')
    for column in ['ptl_side', 'target_current_density_a_m2']:
        if column in accepted and (accepted[column].isna().any() or accepted[column].nunique() != 1):
            raise ValueError(f'Dataset mixes or omits {column}; train on one consistent operating setup')
    if 'ptl_side' in accepted and accepted.ptl_side.iloc[0] != 'anode':
        raise ValueError('This three-variable study expects anode PTL porosity')
    x = accepted[INPUTS].to_numpy(float)
    if np.any(np.ptp(x, axis=0) == 0) or np.linalg.matrix_rank((x-x.mean(axis=0))/x.std(axis=0)) < 3:
        raise ValueError('Training designs must span all three input dimensions')
    if any(np.ptp(accepted[name].to_numpy(float)) == 0 for name in OUTPUTS):
        raise ValueError('Both objectives must vary to fit and validate these normalized GP models')
    return accepted.reset_index(drop=True), excluded


def make_model(seed=42, restarts=3):
    if restarts < 0:
        raise ValueError('--restarts must be nonnegative')
    kernel = (ConstantKernel(1., (1e-3, 1e4)) *
              Matern(length_scale=np.ones(3), length_scale_bounds=(1e-2, 1e3), nu=2.5) +
              WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2)))
    return Pipeline([
        ('scale', StandardScaler()),
        ('gp', GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-10,
                                        n_restarts_optimizer=restarts, random_state=seed)),
    ])


def fit_model(x, y, seed, restarts):
    model = make_model(seed, restarts)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always')
        model.fit(x, y)
    details = {
        'kernel': str(model.named_steps['gp'].kernel_),
        'log_marginal_likelihood': float(model.named_steps['gp'].log_marginal_likelihood_value_),
        'warnings': sorted(set(str(w.message) for w in caught)),
        'input_mean': model.named_steps['scale'].mean_.tolist(),
        'input_scale': model.named_steps['scale'].scale_.tolist(),
        'output_mean': float(model.named_steps['gp']._y_train_mean),
        'output_scale': float(model.named_steps['gp']._y_train_std),
    }
    return model, details


def scores(actual, predicted, std=None):
    residual = predicted-actual
    span = float(np.ptp(actual))
    rmse = float(np.sqrt(np.mean(residual**2)))
    result = {'r2': None if span == 0 else float(1-np.sum(residual**2)/np.sum((actual-actual.mean())**2)),
              'rmse': rmse, 'mae': float(np.mean(np.abs(residual))),
              'max_abs_error': float(np.max(np.abs(residual))),
              'rmse_over_observed_range': None if span == 0 else rmse/span}
    if std is not None:
        result.update(coverage_95=float(np.mean(np.abs(residual) <= 1.96*std)),
                      mean_interval_width_95=float(np.mean(2*1.96*std)))
    return result


def cross_validate(frame, folds=5, seed=42, restarts=3, progress=print):
    if not 2 <= folds <= len(frame)//2:
        raise ValueError('--folds must be between 2 and half the number of samples')
    x, y = frame[INPUTS].to_numpy(float), frame[OUTPUTS].to_numpy(float)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = frame.copy()
    oof['cv_fold'] = -1
    predicted, uncertainty, baseline = (np.empty_like(y) for _ in range(3))
    fold_metrics, diagnostics = [], []
    for fold, (train, test) in enumerate(splitter.split(x), 1):
        oof.loc[test, 'cv_fold'] = fold
        for j, objective in enumerate(OUTPUTS):
            progress(f'CV fold {fold}/{folds}: {objective}', flush=True)
            model, detail = fit_model(x[train], y[train, j], seed+fold*10+j, restarts)
            predicted[test, j], uncertainty[test, j] = model.predict(x[test], return_std=True)
            linear = Pipeline([('scale', StandardScaler()), ('linear', LinearRegression())])
            linear.fit(x[train], y[train, j])
            baseline[test, j] = linear.predict(x[test])
            for label, pred, std in [('gp', predicted[test, j], uncertainty[test, j]),
                                     ('linear_baseline', baseline[test, j], None)]:
                fold_metrics.append({'objective': objective, 'model': label, 'fold': fold,
                                     'train_count': len(train), 'test_count': len(test), 'unit': UNITS[j],
                                     **scores(y[test, j], pred, std)})
            diagnostics.append({'objective': objective, 'fold': fold,
                                'train_indices': train.tolist(), 'test_indices': test.tolist(), **detail})
    metrics = []
    for j, objective in enumerate(OUTPUTS):
        for label, pred, std in [('gp', predicted[:, j], uncertainty[:, j]),
                                 ('linear_baseline', baseline[:, j], None)]:
            metrics.append({'objective': objective, 'model': label, 'unit': UNITS[j],
                            'samples': len(frame), **scores(y[:, j], pred, std)})
        oof[f'{objective}_gp_mean'] = predicted[:, j]
        oof[f'{objective}_gp_std'] = uncertainty[:, j]
        oof[f'{objective}_gp_lower95'] = predicted[:, j]-1.96*uncertainty[:, j]
        oof[f'{objective}_gp_upper95'] = predicted[:, j]+1.96*uncertainty[:, j]
        oof[f'{objective}_linear_baseline'] = baseline[:, j]
    return oof, pd.DataFrame(metrics), pd.DataFrame(fold_metrics), diagnostics


def predict_saved(model_dir, frame, metadata, allow_extrapolation=False):
    missing = set(INPUTS)-set(frame.columns)
    if missing:
        raise ValueError(f'Prediction CSV missing columns: {sorted(missing)}')
    if metadata['inputs'] != INPUTS or metadata['outputs'] != OUTPUTS:
        raise ValueError('Model input/output schema does not match this predictor')
    x = frame[INPUTS].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    if len(x) == 0 or not np.isfinite(x).all() or np.any(x[:, :2] <= 0) or np.any((x[:, 2] <= 0) | (x[:, 2] >= 1)):
        raise ValueError('Prediction inputs must be finite and physically valid')
    outside = np.any((x < metadata['observed_input_min']) | (x > metadata['observed_input_max']), axis=1)
    if outside.any() and not allow_extrapolation:
        raise ValueError('Prediction inputs extend beyond observed training ranges; use --allow-extrapolation explicitly')
    result = frame.copy()
    result['outside_observed_input_ranges'] = outside
    for output in OUTPUTS:
        # Only load artifacts produced by this workflow; joblib is a pickle-based format.
        model = joblib.load(Path(model_dir)/f'{output}.joblib')
        mean, std = model.predict(x, return_std=True)
        result[f'{output}_gp_mean'] = mean
        result[f'{output}_gp_std'] = std
        result[f'{output}_gp_lower95'] = mean-1.96*std
        result[f'{output}_gp_upper95'] = mean+1.96*std
    return result
