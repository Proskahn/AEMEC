"""Train two independent AEMEC GPs and evaluate their out-of-fold predictions."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import tempfile

import joblib
import numpy as np

if __package__:
    from .models import INPUTS, OUTPUTS, load_data, cross_validate, fit_model
else:
    from models import INPUTS, OUTPUTS, load_data, cross_validate, fit_model

ROOT = Path(__file__).resolve().parents[1]


def plot_validation(oof, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Cell voltage [V]', 'H₂ crossover [mol/s]']
    colors = ['#176B9B', '#C45732']
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for j, name in enumerate(OUTPUTS):
        actual = oof[name].to_numpy()
        mean = oof[f'{name}_gp_mean'].to_numpy()
        std = oof[f'{name}_gp_std'].to_numpy()
        ax = axes[0, j]
        ax.errorbar(actual, mean, yerr=1.96*std, fmt='o', ms=3, elinewidth=.6, alpha=.6, color=colors[j])
        extent = [min(actual.min(), (mean-1.96*std).min()), max(actual.max(), (mean+1.96*std).max())]
        ax.plot(extent, extent, color='#777777', ls='--')
        ax.set(xlabel='Observed '+labels[j], ylabel='Held-out GP prediction', title='Mean and nominal 95% interval')
        axes[1, j].scatter(actual, mean-actual, s=14, alpha=.7, c=colors[j])
        axes[1, j].axhline(0, c='#777777', ls='--')
        axes[1, j].set(xlabel='Observed '+labels[j], ylabel='Prediction − observation', title='Held-out residuals')
        for row in range(2):
            axes[row, j].grid(alpha=.2)
            if j == 1:
                axes[row, j].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    fig.suptitle('AEMEC Gaussian-process cross-validation', fontsize=15)
    for extension in ['png', 'pdf']:
        fig.savefig(destination/f'cross_validation.{extension}', dpi=180)
    plt.close(fig)


def train(source, output, expected_samples=256, folds=5, seed=42, restarts=3):
    source, output = Path(source), Path(output)
    frame, excluded = load_data(source, expected_samples)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; choose a new --output to preserve trained results')
    if restarts < 0:
        raise ValueError('--restarts must be nonnegative')
    # Validate before creating directories or starting expensive fits.
    if not 2 <= folds <= len(frame)//2:
        raise ValueError('--folds must be between 2 and half the number of samples')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Failed runs leave no apparently complete model set. Publish only a complete bundle.
    with tempfile.TemporaryDirectory(prefix='.gp-training-', dir=output.parent) as temporary:
        work = Path(temporary)
        oof, metrics, fold_metrics, diagnostics = cross_validate(frame, folds, seed, restarts)
        frame.to_csv(work/'training_data.csv', index=False)
        excluded.to_csv(work/'excluded.csv', index=False)
        oof.to_csv(work/'cv_predictions.csv', index=False)
        metrics.to_csv(work/'cv_metrics.csv', index=False)
        fold_metrics.to_csv(work/'cv_fold_metrics.csv', index=False)
        final = []
        x = frame[INPUTS].to_numpy(float)
        for j, objective in enumerate(OUTPUTS):
            print(f'Final fit on all {len(frame)} samples: {objective}', flush=True)
            model, details = fit_model(x, frame[objective].to_numpy(float), seed+1000+j, restarts)
            joblib.dump(model, work/f'{objective}.joblib')
            final.append({'objective': objective, **details})
        metadata = {
            'schema': 1, 'inputs': INPUTS, 'outputs': OUTPUTS, 'source': str(source.resolve()),
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'sample_count': len(frame), 'excluded_count': len(excluded),
            'observed_input_min': x.min(axis=0).tolist(), 'observed_input_max': x.max(axis=0).tolist(),
            'folds': folds, 'seed': seed, 'optimizer_restarts': restarts,
            'kernel_specification': 'Constant * Matern(nu=2.5, separate length scale per input) + WhiteKernel',
            'kernel_bounds': {'amplitude': [1e-3, 1e4], 'length_scale': [1e-2, 1e3], 'noise': [1e-10, 1e-2]},
            'jitter_alpha': 1e-10, 'cv_input_scaling': 'StandardScaler fit on each training fold',
            'cv_output_scaling': 'normalize_y=True, using only training-fold mean and standard deviation',
            'uncertainty': 'GP predictive standard deviation in physical units, including fitted white-noise kernel; '
                           'conditional on optimized hyperparameters, not calibrated physical-model uncertainty',
            'versions': {name: importlib.metadata.version(name) for name in
                         ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'joblib', 'threadpoolctl']},
            'python_version': platform.python_version(), 'cv_fits': diagnostics, 'final_fits': final,
        }
        (work/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
        plot_validation(oof, work)
        warning_count = sum(len(d['warnings']) for d in diagnostics+final)
        report = [f'# AEMEC Gaussian-process training\n\n{len(frame)} accepted samples, two independent GPs, '
                  f'{folds}-fold shuffled cross-validation (seed {seed}).\n',
                  '| Objective | Model | CV R² | RMSE | MAE | Unit |',
                  '| --- | --- | ---: | ---: | ---: | --- |']
        for row in metrics.to_dict('records'):
            report.append(f"| {row['objective']} | {row['model']} | {row['r2']:.6g} | "
                          f"{row['rmse']:.6g} | {row['mae']:.6g} | {row['unit']} |")
        report += ['', 'Metrics above use pooled held-out predictions, not training predictions. '
                   'Fold metrics, assignments, kernels, and preprocessing parameters are saved.',
                   'Hyperparameters are refit within every fold. Final saved models use all accepted samples.',
                   'Intervals are nominal GP predictive intervals, not guaranteed error bounds. '
                   'Their held-out coverage is reported in cv_metrics.csv. Negative crossover predictions '
                   'are not clipped; inspect such predictions before using the surrogate.',
                   'Random-fold CV measures interpolation over this sampled domain, not extrapolation. '
                   'Kernel/model selection based on these results requires independent or nested validation.',
                   f'Captured optimization warnings: {warning_count}. Inspect metadata.json for boundary or convergence warnings.',
                   'Only load joblib artifacts you trust; use the recorded dependency versions for reproducibility.']
        (work/'report.md').write_text('\n'.join(report)+'\n')
        # rename replaces an existing empty directory on supported POSIX hosts.
        work.rename(output)
    print(metrics.to_string(index=False), flush=True)
    print(f'Saved models and validation report: {output.resolve()}', flush=True)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path, nargs='?', default=ROOT/'ELA/results/evaluated_256.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'GP/results')
    parser.add_argument('--expected-samples', type=int, default=256)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--restarts', type=int, default=3, help='Kernel optimization restarts after the initial optimization')
    args = parser.parse_args(argv)
    try:
        train(args.csv, args.output, args.expected_samples, args.folds, args.seed, args.restarts)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'GP error: {exc}\n')


if __name__ == '__main__':
    main()
