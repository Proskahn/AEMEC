"""Single-point, two-objective EHVI for minimization over nine valid type pairs."""
import itertools
import json

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import qmc

from .data import CATEGORICAL, CONTINUOUS, DESIGN, OBJECTIVES, LOWER, UPPER, check_design
from .surrogate import load_surrogates, predict


def pareto_front(y):
    y = np.unique(np.asarray(y, dtype=float).reshape(-1, 2), axis=0)
    return y[np.array([not np.any(np.all(y <= row, axis=1) & np.any(y < row, axis=1)) for row in y], dtype=bool)]


def positive_part(threshold, mean, std):
    """E[(threshold - Y)+] for a Gaussian, including deterministic predictions."""
    if np.isneginf(threshold):
        return np.zeros_like(mean)
    delta = threshold-mean
    safe_std = np.where(std > 0, std, 1.)
    with np.errstate(over='ignore', under='ignore'):
        z = delta/safe_std
        value = delta*ndtr(z)+std*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    return np.where(std > 0, np.maximum(value, 0.), np.maximum(delta, 0.))


def expected_hypervolume_improvement(means, stds, observed, reference):
    """Integrate independent objective posteriors over non-dominated vertical strips.

    At strip [l,r], the expected covered width is EI(r)-EI(l), and
    the independent expected height is EI(the strip's y ceiling).
    This is for q=1; ranking several points is NOT a joint batch acquisition.
    """
    means, stds, reference = np.asarray(means), np.asarray(stds), np.asarray(reference)
    if (means.ndim != 2 or means.shape[1] != 2 or stds.shape != means.shape or reference.shape != (2,)
            or not np.isfinite(means).all() or not np.isfinite(stds).all() or not np.isfinite(reference).all()
            or np.any(stds < 0)):
        raise ValueError('EHVI expects finite N×2 means/nonnegative standard deviations and a two-value reference')
    front = pareto_front(observed)
    front = front[np.all(front < reference, axis=1)]
    front = front[np.argsort(front[:, 0])]
    score = np.zeros(len(means))
    lower, ceiling = -np.inf, reference[1]
    for row in [*front, reference]:
        upper = row[0]
        width = positive_part(upper, means[:, 0], stds[:, 0])-positive_part(lower, means[:, 0], stds[:, 0])
        score += np.maximum(width, 0.)*positive_part(ceiling, means[:, 1], stds[:, 1])
        lower, ceiling = upper, row[1]
    return score


def not_previously_evaluated(candidates, existing):
    keep = np.ones(len(candidates), dtype=bool)
    candidate_x = (candidates[CONTINUOUS].to_numpy()-LOWER)/(UPPER-LOWER)
    for row in existing.itertuples(index=False):
        same = np.ones(len(candidates), dtype=bool)
        for name in CATEGORICAL:
            same &= candidates[name].to_numpy() == getattr(row, name)
        unit = (np.array([getattr(row, k) for k in CONTINUOUS])-LOWER)/(UPPER-LOWER)
        keep &= ~(same & np.all(np.abs(candidate_x-unit) <= 1e-10, axis=1))
    return candidates.loc[keep].reset_index(drop=True)


def suggest(directory, output, seed=42, candidates_per_combination=512, reference=None, pending=None):
    if candidates_per_combination < 1:
        raise ValueError('--candidates-per-combination must be positive')
    models, representation, metadata = load_surrogates(directory)
    catalog = representation['catalog']
    training = pd.read_csv(directory/'training_data.csv')
    unit = qmc.Sobol(3, scramble=True, seed=seed).random_base2(
        int(np.ceil(np.log2(candidates_per_combination))))[:candidates_per_combination]
    # Include boundary corners and the centre in addition to the finite Sobol pool.
    unit = np.unique(np.vstack([unit, list(itertools.product([0., 1.], repeat=3)), [.5, .5, .5]]), axis=0)
    blocks = []
    for combo in itertools.product(*(catalog['variables'][k]['types'] for k in CATEGORICAL)):
        block = pd.DataFrame(LOWER+unit*(UPPER-LOWER), columns=CONTINUOUS)
        for name, value in zip(CATEGORICAL, combo):
            block[name] = value
        blocks.append(block)
    candidates = pd.concat(blocks, ignore_index=True)
    candidates = not_previously_evaluated(candidates, training)
    pending_frame = None
    if pending is not None:
        pending_frame = check_design(pd.read_csv(pending), catalog)
        candidates = not_previously_evaluated(candidates, pending_frame)
    if candidates.empty:
        raise ValueError('All candidate points are already evaluated or pending; change the seed or pool size')
    means, stds = predict(models, candidates, representation, metadata['encoding'], latent_function=True)
    y = training[OBJECTIVES].to_numpy(float)
    center, scale = y.min(axis=0), np.ptp(y, axis=0)
    reference = y.max(axis=0)+.1*scale if reference is None else np.asarray(reference, dtype=float)
    if reference.shape != (2,) or not np.isfinite(reference).all() or not np.any(np.all(y < reference, axis=1)):
        raise ValueError('Reference must be two finite physical-unit values, worse than at least one observed point in both objectives')
    acquisition = expected_hypervolume_improvement((means-center)/scale, stds/scale,
                                                   (y-center)/scale, (reference-center)/scale)
    candidates['ehvi_normalized'] = acquisition
    for j, name in enumerate(OBJECTIVES):
        candidates[f'{name}_gp_mean'] = means[:, j]
        candidates[f'{name}_gp_latent_std'] = stds[:, j]
    candidates = candidates.sort_values('ehvi_normalized', ascending=False, kind='stable').reset_index(drop=True)
    candidates.to_csv(output/'candidate_scores.csv', index=False)
    suggestion = candidates.iloc[:1].copy()
    next_id = 0
    identifier_frames = [training] + ([pending_frame] if pending_frame is not None else [])
    identifier_columns = [f.sample_id for f in identifier_frames if 'sample_id' in f]
    identifiers = pd.concat(identifier_columns, ignore_index=True) if identifier_columns else pd.Series(dtype=float)
    if len(identifiers):
        numeric_ids = pd.to_numeric(identifiers, errors='coerce').dropna()
        next_id = int(numeric_ids.max())+1 if len(numeric_ids) else len(training)
    suggestion.insert(0, 'sample_id', next_id)
    for name in OBJECTIVES:
        suggestion[name] = np.nan
    suggestion['status'] = 'pending'
    suggestion.to_csv(output/'next_design.csv', index=False)
    pd.DataFrame(pareto_front(y), columns=OBJECTIVES).to_csv(output/'observed_pareto.csv', index=False)
    info = {'acquisition': 'analytic two-objective EHVI; independent latent GP posteriors; q=1; both minimized',
            'reference_physical_units': reference.tolist(), 'objective_center': center.tolist(),
            'objective_scale': scale.tolist(), 'seed': seed, 'candidate_count': len(candidates),
            'candidate_pool': 'all nine category pairs × finite Sobol/corners/centre pool; no global optimum guarantee',
            'source_model_directory': str(directory.resolve()), 'source_representation_sha256': metadata['representation_sha256'],
            'pending_file': str(pending) if pending else None,
            'note': 'Pending points are excluded, not fantasized; use a sequential evaluate/refit/suggest loop. '
                    'GP predictions are not new simulation observations. No material configuration or CFD is executed.'}
    (output/'suggestion.json').write_text(json.dumps(info, indent=2, allow_nan=False)+'\n')
    print(suggestion.to_string(index=False), flush=True)
    return suggestion
