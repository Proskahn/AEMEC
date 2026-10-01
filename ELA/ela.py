"""Offline AEMEC exploratory landscape analysis; never invokes a simulator."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import qmc

INPUTS = ['membrane_thickness_um', 'water_inlet_temperature_k', 'ptl_porosity']
OUTPUTS = ['cell_voltage_v', 'crossover_rate_mol_s']
LOWER = [10., 298.15, .4]
UPPER = [100., 353.15, .8]

# The 49 features in the requested seven-class table, using pflacco's names.
# ICoFiS uses underscores in pflacco: e.g. ic.h_max corresponds to ic.h.max.
DISPERSION_QUANTILES = (.02, .05, .1, .25)
LEVEL_QUANTILES = (.1, .25, .5)
LEVEL_FOLDS = 3
IC_NEIGHBORHOOD = 20
IC_EPSILON = np.insert(10. ** np.linspace(-5, 15, 1000), 0, 0)
FEATURE_NAMES = {
    'distribution': tuple('ela_distr.' + key for key in ('skewness', 'kurtosis', 'number_of_peaks')),
    'level': tuple(f'ela_level.{key}_{q}' for key in ('mmce_lda', 'mmce_qda', 'lda_qda')
                   for q in (10, 25, 50)),
    'meta': tuple('ela_meta.' + key for key in (
        'lin_simple.adj_r2', 'lin_simple.intercept', 'lin_simple.coef.min',
        'lin_simple.coef.max', 'lin_simple.coef.max_by_min', 'lin_w_interact.adj_r2',
        'quad_simple.adj_r2', 'quad_simple.cond', 'quad_w_interact.adj_r2')),
    'dispersion': tuple(f'disp.{key}_{q:02d}' for key in (
        'ratio_mean', 'ratio_median', 'diff_mean', 'diff_median') for q in (2, 5, 10, 25)),
    'nbc': tuple('nbc.' + key for key in (
        'nn_nb.sd_ratio', 'nn_nb.mean_ratio', 'nn_nb.cor', 'dist_ratio.coeff_var', 'nb_fitness.cor')),
    'pca': ('pca.expl_var_PC1.cov_init', 'pca.expl_var_PC1.cor_init'),
    'ic': ('ic.h_max', 'ic.eps_s', 'ic.eps_max', 'ic.eps_ratio', 'ic.m0'),
}
FEATURE_COUNT = sum(map(len, FEATURE_NAMES.values()))


def bounds(lower, upper):
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
        raise ValueError('Require three finite lower/upper bounds with upper > lower')
    return lo, hi


def sample(n=128, seed=42, lower=LOWER, upper=UPPER):
    lo, hi = bounds(lower, upper)
    if n < 1 or n & (n - 1):
        raise ValueError('Sobol sample count must be a positive power of two')
    x = qmc.Sobol(3, scramble=True, seed=seed).random_base2(int(np.log2(n)))
    frame = pd.DataFrame(qmc.scale(x, lo, hi), columns=INPUTS)
    frame.insert(0, 'sample_id', np.arange(n))
    return frame


def prepare(frame, lower=LOWER, upper=UPPER):
    lo, hi = bounds(lower, upper)
    missing = set(INPUTS + OUTPUTS) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    frame = frame.copy()
    reasons = pd.Series('', index=frame.index)
    if 'status' in frame:
        ok = frame.status.astype(str).str.lower().isin(['ok', 'success', 'complete', 'completed'])
        reasons.loc[~ok] = 'unsuccessful_status'
    values = frame[INPUTS + OUTPUTS].apply(pd.to_numeric, errors='coerce')
    bad = ~np.isfinite(values.to_numpy()).all(axis=1)
    reasons.loc[(reasons == '') & bad] = 'missing_or_nonfinite_value'
    invalid = (values.cell_voltage_v <= 0) | (values.crossover_rate_mol_s < 0)
    reasons.loc[(reasons == '') & invalid] = 'invalid_objective'
    outside = ((values[INPUTS] < lo) | (values[INPUTS] > hi)).any(axis=1)
    reasons.loc[(reasons == '') & outside] = 'outside_input_bounds'
    rejected = frame.loc[reasons != ''].copy()
    rejected['exclusion_reason'] = reasons[reasons != '']
    accepted = frame.loc[reasons == ''].copy()
    accepted[INPUTS + OUTPUTS] = values.loc[accepted.index]
    if accepted[INPUTS].duplicated().any():
        raise ValueError('Duplicate successful input designs: aggregate replicates explicitly before ELA')
    if len(accepted) < 32:
        raise ValueError(f'At least 32 valid unique designs required; found {len(accepted)}')
    x = (accepted[INPUTS] - lo) / (hi - lo)
    if np.linalg.matrix_rank(x.to_numpy() - x.to_numpy().mean(axis=0)) < 3:
        raise ValueError('Accepted designs do not span all three input dimensions')
    return accepted.reset_index(drop=True), rejected, x.reset_index(drop=True)


def pareto_mask(y):
    y = np.asarray(y)
    return np.array([not np.any(np.all(y <= p, axis=1) & np.any(y < p, axis=1)) for p in y])


def features(x, y, seed=42, ic_sorting='nn'):
    """Compute the table's 49 features on already normalized inputs and output.

    Undefined/failed features retain their names with value None. Call analyze()
    for CSV validation and normalization. Neither path evaluates the simulator.
    """
    from pflacco import classical_ela_features as pf
    if ic_sorting not in ('nn', 'random'):
        raise ValueError('IC sorting must be nn or random')
    y = np.asarray(y, dtype=float)
    families = {
        'distribution': (pf.calculate_ela_distribution, {}),
        'level': (pf.calculate_ela_level, {'ela_level_quantiles': list(LEVEL_QUANTILES),
                                         'ela_level_resample_iterations': LEVEL_FOLDS}),
        'meta': (pf.calculate_ela_meta, {}),
        'dispersion': (pf.calculate_dispersion, {'disp_quantiles': list(DISPERSION_QUANTILES),
                                               'minimize': True}),
        'nbc': (pf.calculate_nbc, {'dist_tie_breaker': 'first', 'minimize': True}),
        'pca': (pf.calculate_pca, {}),
        'ic': (pf.calculate_information_content, {
            'ic_sorting': ic_sorting, 'ic_nn_neighborhood': IC_NEIGHBORHOOD,
            'ic_epsilon': IC_EPSILON, 'ic_settling_sensitivity': .05,
            'ic_info_sensitivity': .5, 'seed': seed}),
    }
    result = {key: None for keys in FEATURE_NAMES.values() for key in keys}
    diagnostics = []
    if np.ptp(y) == 0:
        return result, [{'family': name, 'status': 'skipped',
                         'message': 'Constant output; feature computation skipped',
                         'undefined_features': list(keys)} for name, keys in FEATURE_NAMES.items()]
    state = np.random.get_state()
    try:
        for name, (function, kwargs) in families.items():
            expected = FEATURE_NAMES[name]
            details = {}
            skipped_keys = set()
            if name == 'dispersion':
                counts = {f'{q:.2f}': int(np.count_nonzero(y <= np.quantile(y, q)))
                          for q in DISPERSION_QUANTILES}
                details['subset_counts'] = counts
                kwargs['disp_quantiles'] = [q for q in DISPERSION_QUANTILES if counts[f'{q:.2f}'] >= 2]
                for q in DISPERSION_QUANTILES:
                    if counts[f'{q:.2f}'] < 2:
                        skipped_keys.update(k for k in expected if k.endswith(f'_{round(q*100):02d}'))
                if skipped_keys:
                    diagnostics.append({'family': name, 'status': 'warning',
                                        'message': 'Dispersion needs at least two points in each quantile subset; '
                                                   'undersized subsets are left undefined.',
                                        'undefined_features': sorted(skipped_keys)})
            np.random.seed(seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    values = function(x.copy(), y.copy(), **kwargs)
                    missing = set(expected) - skipped_keys - values.keys()
                    if missing:
                        raise ValueError(f'pflacco did not return expected features: {sorted(missing)}')
                    # Filter extra PCA features and runtimes to retain exactly the table's 49.
                    for key in expected:
                        value = values.get(key)
                        result[key] = float(value) if value is not None and np.isfinite(value) else None
                    undefined = [k for k in expected if result[k] is None]
                    diagnostics.append({'family': name, 'status': 'partial' if undefined else 'ok',
                                        'undefined_features': undefined, **details})
                except Exception as exc:
                    for key in expected:
                        result[key] = None
                    diagnostics.append({'family': name, 'status': 'failed', 'message': str(exc),
                                        'undefined_features': list(expected), **details})
                for warning in caught:
                    diagnostics.append({'family': name, 'status': 'warning', 'message': str(warning.message)})
    finally:
        np.random.set_state(state)
    return result, diagnostics


def analyze(source, destination, lower=LOWER, upper=UPPER, seed=42, scales=None, provenance='unknown',
            ic_sorting='nn'):
    source, destination = Path(source), Path(destination)
    frame = pd.read_csv(source)
    accepted, rejected, x = prepare(frame, lower, upper)
    y = accepted[OUTPUTS].to_numpy(float)
    if scales is None:
        limits = np.array([y.min(axis=0), y.max(axis=0)])
    else:
        limits = np.asarray(scales, float).reshape(2, 2).T
        if not np.isfinite(limits).all() or np.any(limits[1] <= limits[0]):
            raise ValueError('Objective scales require finite min/max pairs with max > min')
    span = limits[1] - limits[0]
    z = (y - limits[0]) / np.where(span == 0, 1, span)
    targets = {name: z[:, i] for i, name in enumerate(OUTPUTS)}
    targets.update({f'weighted_voltage_{w:g}': w*z[:, 0] + (1-w)*z[:, 1] for w in [.25, .5, .75]})
    rows, diagnostics = [], {}
    for name, target in targets.items():
        values, diagnostics[name] = features(x, target, seed, ic_sorting)
        rows.extend({'landscape': name, 'feature': k, 'value': v} for k, v in values.items())
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=['landscape', 'feature', 'value']).to_csv(destination/'features.csv', index=False)
    accepted.to_csv(destination/'accepted.csv', index=False)
    rejected.to_csv(destination/'excluded.csv', index=False)
    mask = pareto_mask(y)
    accepted.loc[mask].to_csv(destination/'pareto.csv', index=False)
    corr = accepted[INPUTS + OUTPUTS].corr(method='spearman')
    corr.to_csv(destination/'spearman.csv')
    notes = ['Features describe the accepted sample, not a proof of convexity or a count of local optima.',
             'Both outputs are minimized. Feature outputs use affine objective normalization.',
             '49 features per landscape across seven classes; undefined values remain blank.',
             'Level-set features use 3 folds; dispersion uses 2%, 5%, 10% and 25% subsets.',
             'PCA uses the augmented normalized input-objective matrix [X, y].',
             f'Information content uses a seeded {ic_sorting} tour through existing samples; no new CFD calls.']
    if provenance != 'space-filling':
        notes.append('Sampling is adaptive or unknown: global landscape interpretation may be biased.')
    if len(rejected):
        notes.append('Excluded evaluations restrict the landscape to the successfully evaluated region.')
    if np.any(span == 0):
        notes.append('A constant output was mapped to zero; its standalone feature values are blank.')
    meta = {'source': str(source.resolve()), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'seed': seed, 'sampling': provenance, 'input_columns': INPUTS,
            'input_lower': list(lower), 'input_upper': list(upper),
            'objective_columns': OUTPUTS, 'objective_limits': limits.tolist(),
            'scale_source': 'sample_min_max' if scales is None else 'explicit',
            'attempts': len(frame), 'accepted': len(accepted), 'excluded': len(rejected),
            'pareto_count': int(mask.sum()), 'notes': notes, 'diagnostics': diagnostics,
            'feature_set': {'name': 'seven_class_49', 'count_per_landscape': FEATURE_COUNT,
                            'families': FEATURE_NAMES},
            'feature_settings': {
                'level_quantiles': LEVEL_QUANTILES, 'level_folds': LEVEL_FOLDS,
                'dispersion_quantiles': DISPERSION_QUANTILES, 'nbc_tie_breaker': 'first',
                'ic_sorting': ic_sorting, 'ic_nn_neighborhood': IC_NEIGHBORHOOD,
                'ic_seed': seed, 'ic_epsilon': IC_EPSILON.tolist(),
                'ic_settling_sensitivity': .05, 'ic_info_sensitivity': .5},
            'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'pflacco']}}
    (destination/'metadata.json').write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, column in zip(axes, INPUTS):
        dots = ax.scatter(y[:, 0], y[:, 1], c=accepted[column], cmap='viridis')
        ax.scatter(y[mask, 0], y[mask, 1], facecolors='none', edgecolors='black', s=70)
        ax.set(xlabel='Cell voltage [V]', ylabel='H2 crossover [mol/s]')
        fig.colorbar(dots, ax=ax, label=column)
    fig.savefig(destination/'pareto.png', dpi=160)
    plt.close(fig)
    failures = sum(d['status'] == 'failed' for ds in diagnostics.values() for d in ds)
    undefined_count = sum(row['value'] is None for row in rows)
    (destination/'report.md').write_text(
        '# AEMEC exploratory landscape analysis\n\n'
        f'{len(accepted)} accepted / {len(frame)} rows; {mask.sum()} sampled nondominated designs.\n\n'
        + '\n'.join(f'- {note}' for note in notes)
        + f'\n\nFeature-family failures: {failures}. Inspect metadata.json for warnings and undefined features.\n'
        + f'\nExported {len(rows)} feature rows ({FEATURE_COUNT} per landscape); '
          f'{undefined_count} undefined values.\n'
        + '\nfeatures.csv contains each objective and three weighted combinations; weights are voltage weights. '
        'Normalization endpoints are recorded in metadata.json. Use identical explicit scales for comparisons.\n')
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    for name in ['sample', 'analyze']:
        p = subs.add_parser(name)
        p.add_argument('--lower', nargs=3, type=float, default=LOWER)
        p.add_argument('--upper', nargs=3, type=float, default=UPPER)
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--output', type=Path, required=True)
        if name == 'sample':
            p.add_argument('--n', type=int, default=128)
        else:
            p.add_argument('csv', type=Path)
            p.add_argument('--sampling', choices=['space-filling', 'adaptive', 'unknown'], default='unknown')
            p.add_argument('--objective-scales', nargs=4, type=float, metavar=('V_MIN', 'V_MAX', 'C_MIN', 'C_MAX'))
            p.add_argument('--ic-sorting', choices=['nn', 'random'], default='nn',
                           help='Information-content tour: nearest-neighbor (default) or random permutation')
    args = parser.parse_args()
    try:
        if args.command == 'sample':
            frame = sample(args.n, args.seed, args.lower, args.upper)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.output, index=False)
            args.output.with_suffix('.json').write_text(json.dumps(
                {'method': 'scrambled_sobol', 'n': args.n, 'seed': args.seed,
                 'inputs': INPUTS, 'lower': args.lower, 'upper': args.upper}, indent=2)+'\n')
            print(f'Wrote {len(frame)} unevaluated designs to {args.output}')
        else:
            meta = analyze(args.csv, args.output, args.lower, args.upper, args.seed,
                           args.objective_scales, args.sampling, args.ic_sorting)
            print(f"Analyzed {meta['accepted']} designs; report: {args.output / 'report.md'}")
            for note in meta['notes']:
                print(note)
    except (ValueError, ImportError, OSError) as exc:
        parser.exit(2, f'ELA error: {exc}\n')


if __name__ == '__main__':
    main()
