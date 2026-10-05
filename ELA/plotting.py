"""Two-dimensional sample plots and conditional polynomial response slices."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

if __package__:
    from .ela import INPUTS, OUTPUTS, LOWER, UPPER, bounds, prepare
else:
    from ela import INPUTS, OUTPUTS, LOWER, UPPER, bounds, prepare

INPUT_LABELS = ['Membrane thickness [µm]', 'Water inlet temperature [K]', 'Anode PTL porosity [−]']
OUTPUT_LABELS = ['Cell voltage [V]', 'H₂ crossover [mol/s]']
COLORS = ['#176B9B', '#C45732']


def design_matrix(x, model='quadratic'):
    """Standard total-degree <= 2 polynomial: 10 terms in three variables."""
    x = np.asarray(x, dtype=float)
    if model not in ('linear', 'quadratic'):
        raise ValueError('Model must be linear or quadratic')
    terms = [np.ones(len(x)), *x.T]
    names = ['intercept', *INPUTS]
    if model == 'quadratic':
        terms.extend(x[:, j]**2 for j in range(3))
        names.extend(f'{name}^2' for name in INPUTS)
        for j, k in [(0, 1), (0, 2), (1, 2)]:
            terms.append(x[:, j]*x[:, k])
            names.append(f'{INPUTS[j]} * {INPUTS[k]}')
    return np.column_stack(terms), names


def fit(x, y, model):
    matrix, _ = design_matrix(x, model)
    coefficient, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
    if rank != matrix.shape[1]:
        raise ValueError('Sample cannot identify all polynomial terms; try --model linear or a fuller design')
    # Keep truly constant outputs exactly constant despite floating-point regression noise.
    for j in range(y.shape[1]):
        if np.ptp(y[:, j]) == 0:
            coefficient[:, j] = 0
            coefficient[0, j] = y[0, j]
    return coefficient


def predict(x, coefficient, model):
    return design_matrix(x, model)[0] @ coefficient


def cross_validate(x, y, model, folds=5, seed=42):
    if not 2 <= folds <= len(x):
        raise ValueError('--cv-folds must be between 2 and the accepted sample count')
    prediction = np.empty_like(y)
    fold_ids = np.empty(len(x), dtype=int)
    for fold, test in enumerate(np.array_split(np.random.default_rng(seed).permutation(len(x)), folds)):
        train = np.ones(len(x), dtype=bool)
        train[test] = False
        coefficient = fit(x[train], y[train], model)
        prediction[test] = predict(x[test], coefficient, model)
        fold_ids[test] = fold
    rows = []
    for j, output in enumerate(OUTPUTS):
        residual = prediction[:, j]-y[:, j]
        span = float(np.ptp(y[:, j]))
        rmse = float(np.sqrt(np.mean(residual**2)))
        rows.append({'objective': output, 'model': model, 'cv_folds': folds,
                     'cv_r2': None if span == 0 else float(1-np.sum(residual**2)/np.sum((y[:, j]-y[:, j].mean())**2)),
                     'cv_rmse': rmse, 'cv_mae': float(np.mean(np.abs(residual))),
                     'cv_max_abs_error': float(np.max(np.abs(residual))),
                     'observed_range': span, 'cv_rmse_over_observed_range': None if span == 0 else rmse/span})
    return prediction, fold_ids, rows


def conditional_slices(x, y, coefficient, model, reference, grid_points):
    """Return fixed-reference slices and points adjusted for the other inputs.

    Adjusted point = observed y - fitted y at original x + fitted y with
    non-plotted inputs replaced by their reference values. These are derived
    points, not new CFD observations at the reference conditions.
    """
    fitted = predict(x, coefficient, model)
    result = []
    for j in range(3):
        grid = np.linspace(x[:, j].min(), x[:, j].max(), grid_points)
        section = np.tile(reference, (grid_points, 1))
        section[:, j] = grid
        adjusted_x = np.tile(reference, (len(x), 1))
        adjusted_x[:, j] = x[:, j]
        result.append((grid, predict(section, coefficient, model),
                       y-fitted+predict(adjusted_x, coefficient, model)))
    return result


def correlation(a, b, ranked=False):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    if ranked:
        a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


def direction(curve, objective_range):
    """Numerical sign summary; its tolerance is not a physical significance test."""
    tolerance = max(float(objective_range), float(np.ptp(curve))) * 1e-8
    if np.ptp(curve) <= tolerance:
        return 'flat'
    differences = np.diff(curve)
    if np.all(differences >= -tolerance):
        return 'increasing'
    if np.all(differences <= tolerance):
        return 'decreasing'
    return 'non_monotonic'


def plot_effects(source, destination, lower=LOWER, upper=UPPER, model='quadratic',
                 reference=None, grid_points=201, cv_folds=5, seed=42):
    if grid_points < 3:
        raise ValueError('--grid-points must be at least 3')
    source, destination = Path(source), Path(destination)
    lo, hi = bounds(lower, upper)
    frame, excluded, normalized = prepare(pd.read_csv(source), lower, upper)
    # Reject inconsistent known operating conditions rather than merging different landscapes.
    for column in ['ptl_side', 'target_current_density_a_m2']:
        if column in frame and (frame[column].isna().any() or frame[column].nunique() != 1):
            raise ValueError(f'Input CSV must have one consistent {column}')
    if 'ptl_side' in frame and str(frame.ptl_side.iloc[0]) != 'anode':
        raise ValueError('These plots are for anode PTL porosity; supply an anode study')
    physical = frame[INPUTS].to_numpy(float)
    ref = np.median(physical, axis=0) if reference is None else np.asarray(reference, float)
    if ref.shape != (3,) or not np.isfinite(ref).all() or np.any(ref < physical.min(axis=0)) or np.any(ref > physical.max(axis=0)):
        raise ValueError('--reference must contain three finite values inside the observed input ranges')
    x, y = normalized.to_numpy(float), frame[OUTPUTS].to_numpy(float)
    normalized_ref = (ref-lo)/(hi-lo)
    coefficient = fit(x, y, model)
    cv_prediction, fold_ids, metrics = cross_validate(x, y, model, cv_folds, seed)
    slices = conditional_slices(x, y, coefficient, model, normalized_ref, grid_points)
    spans = np.ptp(y, axis=0)
    baseline = predict(normalized_ref[None, :], coefficient, model)[0]
    summary, curves = [], []
    for j, (grid, prediction, adjusted) in enumerate(slices):
        for k, output in enumerate(OUTPUTS):
            span = float(spans[k])
            delta = float(prediction[-1, k]-prediction[0, k])
            effect_span = float(np.ptp(prediction[:, k]))
            summary.append({'input': INPUTS[j], 'objective': output,
                            'raw_pearson': correlation(physical[:, j], y[:, k]),
                            'raw_spearman': correlation(physical[:, j], y[:, k], ranked=True),
                            'conditional_direction': direction(prediction[:, k], span),
                            'conditional_endpoint_change': delta,
                            'endpoint_change_over_objective_range': None if span == 0 else delta/span,
                            'conditional_effect_span': effect_span, 'cv_rmse': metrics[k]['cv_rmse'],
                            'effect_span_exceeds_cv_rmse': effect_span > metrics[k]['cv_rmse']})
            for value, predicted in zip(grid*(hi[j]-lo[j])+lo[j], prediction[:, k]):
                curves.append({'input': INPUTS[j], 'objective': output, 'input_value': value,
                               'predicted_value': predicted,
                               'change_over_objective_range': None if span == 0 else (predicted-baseline[k])/span})
    notes = [
        'Dots in raw plots are CFD samples; all other inputs vary between dots.',
        'Conditional curves are polynomial predictions with other inputs fixed at the recorded reference.',
        'Adjusted points are model-derived, not additional CFD samples at the reference conditions.',
        'Both objectives are minimized. Opposite curve directions suggest a conditional trade-off.',
        'Same-sign directions suggest aligned effects; nonmonotonic curves require examining local slopes.',
        'Effect directions are numerical summaries, not significance tests or proof of strict monotonicity.',
        'A small effect relative to validation error is difficult to resolve with this surrogate.',
        'Random-fold CV evaluates overall predictive fit, not accuracy of each conditional effect.',
        'Curves stay within observed marginal input ranges but may cross sparsely sampled or infeasible combinations.',
        'No predictions are clipped: negative predicted crossover indicates a surrogate limitation.',
    ]
    if len(excluded):
        notes.append(f'{len(excluded)} rows were excluded; see excluded.csv. Conclusions concern accepted designs.')
    if any(np.any(prediction[:, 0] <= 0) or np.any(prediction[:, 1] < 0) for _, prediction, _ in slices):
        notes.append('Some conditional predictions are nonphysical; inspect the curves and validation before interpreting them.')
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination/'accepted.csv', index=False)
    excluded.to_csv(destination/'excluded.csv', index=False)
    pd.DataFrame(metrics).to_csv(destination/'model_validation.csv', index=False)
    pd.DataFrame(summary).to_csv(destination/'effect_summary.csv', index=False)
    pd.DataFrame(curves).to_csv(destination/'conditional_curves.csv', index=False)
    _, names = design_matrix(x, model)
    pd.DataFrame([{'objective': output, 'term': name, 'coefficient': coefficient[i, k]}
                  for k, output in enumerate(OUTPUTS) for i, name in enumerate(names)]).to_csv(
                      destination/'model_coefficients.csv', index=False)
    predictions = frame.copy()
    predictions['cv_fold'] = fold_ids
    for k, output in enumerate(OUTPUTS):
        predictions[f'cv_predicted_{output}'] = cv_prediction[:, k]
    predictions.to_csv(destination/'cv_predictions.csv', index=False)
    metadata = {'source': str(source.resolve()), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'accepted': len(frame), 'excluded': len(excluded), 'input_lower': lo.tolist(),
                'input_upper': hi.tolist(), 'reference': dict(zip(INPUTS, ref.tolist())),
                'reference_source': 'sample_median' if reference is None else 'explicit',
                'model': model, 'model_terms': names, 'input_normalization': '(x-lower)/(upper-lower)',
                'fitted_output_scale': 'physical units', 'objective_ranges': dict(zip(OUTPUTS, spans.tolist())),
                'cv_seed': seed, 'cv_folds': cv_folds, 'grid_points': grid_points,
                'design_matrix_condition_number': float(np.linalg.cond(design_matrix(x, model)[0])),
                'notes': notes,
                'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'matplotlib']}}
    (destination/'plot_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    render(physical, y, slices, cv_prediction, metrics, ref, baseline, spans, lo, hi, model, destination)
    validation_lines = '\n'.join(
        f"- {r['objective']}: CV R²={r['cv_r2']:.5f}, RMSE={r['cv_rmse']:.6g}" if r['cv_r2'] is not None
        else f"- {r['objective']}: constant output; CV R² undefined" for r in metrics)
    (destination/'plot_report.md').write_text(
        '# AEMEC input–objective plots\n\n'
        f'{len(frame)} accepted designs; {model} regression; {cv_folds}-fold cross-validation.\n\n'
        'Reference values: '+', '.join(f'{name}={v:.6g}' for name, v in zip(INPUTS, ref))+'\n\n'
        +validation_lines+'\n\n'+'\n'.join('- '+note for note in notes)+'\n\n'
        'Open `raw_scatter.png` for observations, `conditional_effects.png` for isolated fitted effects, '
        '`objective_directions.png` to compare the two objectives, and `model_validation.png` for held-out predictions. '
        'PDF versions and numerical CSV tables accompany the figures.\n')
    return metadata


def render(physical, y, slices, cv_prediction, metrics, ref, baseline, spans, lo, hi, model, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def save(fig, name):
        fig.savefig(destination/f'{name}.png', dpi=180, facecolor='white')
        fig.savefig(destination/f'{name}.pdf', facecolor='white')
        plt.close(fig)

    def style(ax, j, k):
        ax.set_xlabel(INPUT_LABELS[j])
        ax.set_ylabel(OUTPUT_LABELS[k])
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        if k == 1:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    with plt.rc_context({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10}):
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)
        for k in range(2):
            for j in range(3):
                ax = axes[k, j]
                ax.scatter(physical[:, j], y[:, k], s=23, c=COLORS[k], alpha=.7, linewidths=0)
                style(ax, j, k)
        fig.suptitle('Observed input–objective relationships\nEach dot is a CFD sample; the other two inputs also vary.', fontsize=14)
        save(fig, 'raw_scatter')

        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), constrained_layout=True)
        reference_labels = [f'Thickness = {ref[0]:.4g} µm', f'Inlet T = {ref[1]:.4g} K',
                            f'Porosity = {ref[2]:.4g}']
        for j, (grid, prediction, adjusted) in enumerate(slices):
            reference_label = '; '.join(reference_labels[i] for i in range(3) if i != j)
            for k in range(2):
                ax = axes[k, j]
                ax.scatter(physical[:, j], adjusted[:, k], s=17, color=COLORS[k], alpha=.4, linewidths=0)
                ax.plot(grid*(hi[j]-lo[j])+lo[j], prediction[:, k], color=COLORS[k], lw=2)
                ax.axvline(ref[j], color='#89929C', ls=':', lw=1)
                ax.set_title('Other inputs fixed:\n'+reference_label, fontsize=9)
                style(ax, j, k)
        fig.legend(handles=[Line2D([], [], color='#333333', lw=2, label=f'{model.capitalize()} fitted slice'),
                            Line2D([], [], color='#777777', marker='o', ls='', label='Adjusted sample (model-derived)')],
                   loc='outside lower center', ncol=2, frameon=False)
        fig.suptitle('Conditional effects on each objective\nDots are adjusted using the fit; these are not new fixed-input CFD runs.', fontsize=14)
        save(fig, 'conditional_effects')

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)
        for j, (grid, prediction, _) in enumerate(slices):
            ax = axes[j]
            for k in range(2):
                # Constant outputs have zero fitted change; mark their scale explicitly in metadata.
                change = (prediction[:, k]-baseline[k])/spans[k] if spans[k] > 0 else np.zeros(len(grid))
                ax.plot(grid*(hi[j]-lo[j])+lo[j], change, c=COLORS[k], lw=2,
                        ls='-' if k == 0 else '--', label=OUTPUT_LABELS[k].split(' [')[0])
            ax.axhline(0, color='#89929C', lw=.8)
            ax.axvline(ref[j], color='#89929C', ls=':', lw=1)
            ax.set_xlabel(INPUT_LABELS[j])
            ax.set_ylabel('Change / observed objective range')
            ax.grid(alpha=.2)
            ax.legend(frameon=False)
        fig.suptitle('Do the objectives move together or in opposite directions?\nChanges relative to the reference design; lower is better for both.', fontsize=14)
        save(fig, 'objective_directions')

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
        for k, ax in enumerate(axes):
            ax.scatter(y[:, k], cv_prediction[:, k], c=COLORS[k], s=23, alpha=.7, linewidths=0)
            extent = [min(y[:, k].min(), cv_prediction[:, k].min()), max(y[:, k].max(), cv_prediction[:, k].max())]
            ax.plot(extent, extent, color='#777777', ls='--', lw=1)
            score = metrics[k]['cv_r2']
            ax.set_title(f"CV R² = {score:.4f}" if score is not None else 'Constant output: R² undefined')
            ax.set_xlabel('Observed '+OUTPUT_LABELS[k])
            ax.set_ylabel('Held-out prediction')
            ax.grid(alpha=.2)
            if k == 1:
                ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
        fig.suptitle('Surrogate validation: predictions for held-out samples', fontsize=14)
        save(fig, 'model_validation')
