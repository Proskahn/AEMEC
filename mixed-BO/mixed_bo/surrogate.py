"""Independent GP objectives on continuous inputs plus frozen material features."""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold, LeaveOneGroupOut
from threadpoolctl import threadpool_limits

from .data import CATEGORICAL, OBJECTIVES, read_evaluations, fingerprint
from .representation import feature_matrix


def fit_gp(x, y, seed=42, restarts=2):
    kernel = (ConstantKernel(1., (1e-3, 1e4))*Matern(np.ones(x.shape[1]), (1e-2, 1e3), nu=2.5)
              + WhiteKernel(1e-6, (1e-10, 1e-1)))
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-10,
                                    n_restarts_optimizer=restarts, random_state=seed)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always')
        model.fit(x, y)
    diagnostics = {'kernel': str(model.kernel_), 'warnings': sorted({str(w.message) for w in caught}),
                   'output_mean': float(model._y_train_mean), 'output_scale': float(model._y_train_std)}
    return model, diagnostics


def metric(actual, predicted, std):
    error = predicted-actual
    span = np.ptp(actual)
    return {'r2': float(1-np.sum(error**2)/np.sum((actual-actual.mean())**2)) if span else None,
            'rmse': float(np.sqrt(np.mean(error**2))), 'mae': float(np.mean(np.abs(error))),
            'rmse_over_range': float(np.sqrt(np.mean(error**2))/span) if span else None,
            'coverage_95': float(np.mean(np.abs(error) <= 1.96*std))}


def splits_for(frame, mode, folds, seed):
    if mode == 'random':
        if not 2 <= folds <= len(frame)//2:
            raise ValueError('CV folds must be between 2 and half the valid sample count')
        return list(KFold(folds, shuffle=True, random_state=seed).split(frame))
    if mode == 'combination':
        groups = pd.MultiIndex.from_frame(frame[CATEGORICAL]).factorize()[0]
    elif mode in CATEGORICAL:
        groups = pd.factorize(frame[mode])[0]
    else:
        raise ValueError('Unknown CV mode')
    if len(np.unique(groups)) < 2:
        raise ValueError('Leave-one-group-out CV requires observations in at least two groups')
    splits = list(LeaveOneGroupOut().split(frame, groups=groups))
    if any(len(train) < 4 for train, _ in splits):
        raise ValueError('Each group-held-out fold needs at least four training observations')
    return splits


def train_surrogates(source, representation, output, encoding='vae', folds=5,
                     cv_mode='random', seed=42, restarts=2):
    if restarts < 0:
        raise ValueError('GP optimization restarts must be nonnegative')
    frame, excluded = read_evaluations(source, representation['catalog'])
    x = feature_matrix(frame, representation, encoding)
    y = frame[OBJECTIVES].to_numpy(float)
    splits = splits_for(frame, cv_mode, folds, seed)
    oof = frame.copy()
    oof['cv_fold'] = -1
    means, stds = np.empty_like(y), np.empty_like(y)
    diagnostics, fold_metrics, final_models = [], [], []
    for fold, (training, test) in enumerate(splits, 1):
        oof.loc[test, 'cv_fold'] = fold
        for j, objective in enumerate(OBJECTIVES):
            print(f'GP CV {fold}/{len(splits)}: {objective}', flush=True)
            model, detail = fit_gp(x[training], y[training, j], seed+fold*10+j, restarts)
            means[test, j], stds[test, j] = model.predict(x[test], return_std=True)
            diagnostics.append({'objective': objective, 'fold': fold, 'train_indices': training.tolist(),
                                'test_indices': test.tolist(), **detail})
            fold_metrics.append({'objective': objective, 'fold': fold, 'train_count': len(training),
                                 'test_count': len(test), **metric(y[test, j], means[test, j], stds[test, j])})
    pooled = []
    for j, objective in enumerate(OBJECTIVES):
        oof[f'{objective}_gp_mean'], oof[f'{objective}_gp_std'] = means[:, j], stds[:, j]
        pooled.append({'objective': objective, 'encoding': encoding, 'samples': len(frame),
                       **metric(y[:, j], means[:, j], stds[:, j])})
        model, detail = fit_gp(x, y[:, j], seed+1000+j, restarts)
        joblib.dump(model, output/f'{objective}.joblib')
        final_models.append({'objective': objective, **detail})
    frame.to_csv(output/'training_data.csv', index=False)
    excluded.to_csv(output/'excluded.csv', index=False)
    oof.to_csv(output/'cv_predictions.csv', index=False)
    pd.DataFrame(pooled).to_csv(output/'cv_metrics.csv', index=False)
    pd.DataFrame(fold_metrics).to_csv(output/'cv_fold_metrics.csv', index=False)
    coverage = {k: {t: int((frame[k] == t).sum()) for t in representation['catalog']['variables'][k]['types']}
                for k in CATEGORICAL}
    notes = ['VAE/catalog features are fitted only to human descriptors and frozen before GP fitting; '
             'GP output normalization and hyperparameter fits use training-fold observations only.',
             'GP predictive intervals condition on the fitted embedding and kernel; VAE uncertainty is not propagated.',
             'Gaussian objective predictions can be negative and are not clipped; inspect extrapolative predictions.',
             'Descriptor reconstruction alone does not guarantee preservation of material distances or BO improvement.',
             'Random CV estimates within-catalog prediction. Group-held-out CV tests transfer across specified groups.']
    for k, counts in coverage.items():
        if any(n == 0 for n in counts.values()):
            notes.append(f'{k} has unobserved types: predictions there rely on representation-based transfer, '
                         'which has not been established by observations of those types.')
    warning_count = sum(len(d['warnings']) for d in diagnostics+final_models)
    notes.append(f'Captured GP optimization warnings: {warning_count}; details are recorded in metadata.json.')
    if representation['catalog'].get('synthetic_demo'):
        notes.append('SYNTHETIC DEMONSTRATION: descriptors/objectives are not AEMEC material measurements.')
    metadata = {'schema': 1, 'encoding': encoding, 'sample_count': len(frame), 'cv_mode': cv_mode,
                'cv_folds': len(splits), 'seed': seed, 'restarts': restarts, 'category_coverage': coverage,
                'representation_sha256': fingerprint(representation), 'cv_fits': diagnostics,
                'final_fits': final_models, 'notes': notes}
    (output/'representation.json').write_text(json.dumps(representation, indent=2, allow_nan=False)+'\n')
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    report = ['# Mixed-variable GP validation', '',
              f'{len(frame)} observations; encoding `{encoding}`; CV `{cv_mode}` ({len(splits)} folds).', '',
              '| Objective | CV R² | RMSE | MAE |', '| --- | ---: | ---: | ---: |']
    for row in pooled:
        report.append(f"| {row['objective']} | {row['r2']:.6g} | {row['rmse']:.6g} | {row['mae']:.6g} |")
    report += ['', *notes]
    (output/'report.md').write_text('\n'.join(report)+'\n')
    print(pd.DataFrame(pooled).to_string(index=False), flush=True)
    return metadata


def load_surrogates(directory):
    from .representation import load_representation
    directory = Path(directory)
    representation = load_representation(directory)
    metadata = json.loads((directory/'metadata.json').read_text())
    if metadata['representation_sha256'] != fingerprint(representation):
        raise ValueError('The representation changed after GP training; retrain both GPs')
    models = [joblib.load(directory/f'{name}.joblib') for name in OBJECTIVES]
    return models, representation, metadata


def predict(models, frame, representation, encoding, latent_function=False):
    x = feature_matrix(frame, representation, encoding)
    means, stds = [], []
    with threadpool_limits(limits=1):
        for model in models:
            mean, std = model.predict(x, return_std=True)
            if latent_function:
                noise_variance = model.kernel_.k2.noise_level*float(model._y_train_std)**2
                std = np.sqrt(np.maximum(0., std**2-noise_variance))
            means.append(mean)
            stds.append(std)
    return np.column_stack(means), np.column_stack(stds)
