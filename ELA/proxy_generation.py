"""Landscape-guided genetic programming, following sections 3.1/3.2 and 5.3.

This independent implementation uses the existing 49-feature ELA pipeline.
Implementation choices not specified by the paper are recorded in the README
and run manifest, notably per-feature Wasserstein aggregation and normalization.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc, wasserstein_distance

if __package__:
    from .ela import FEATURE_NAMES, INPUTS, OUTPUTS, features, prepare
    from .proxy_functions import ARITY, Node, ProxyFunction, evaluate_tree
else:
    from ela import FEATURE_NAMES, INPUTS, OUTPUTS, features, prepare
    from proxy_functions import ARITY, Node, ProxyFunction, evaluate_tree

FEATURES = tuple(k for family in FEATURE_NAMES.values() for k in family)
PAPER = 'https://arxiv.org/abs/2602.04529v1'


def save_json(path, value):
    """Replace a JSON checkpoint only after the complete document is written."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_analysis(directory, landscapes):
    directory = Path(directory)
    paths = {name: directory/name for name in ('features.csv', 'accepted.csv', 'metadata.json')}
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f'Missing {path}. First run ELA/ela.py analyze; a single feature '
                             'vector cannot reconstruct the repeated-subset distributions.')
    meta = json.loads(paths['metadata.json'].read_text())
    if meta.get('input_columns') != INPUTS or meta.get('objective_columns') != OUTPUTS:
        raise ValueError('Analysis input/objective schema does not match this AEMEC pipeline')
    if meta.get('feature_set', {}).get('count_per_landscape') != len(FEATURES):
        raise ValueError('Regenerate the analysis using the current 49-feature ELA implementation')
    # Preserve the original analysis floats: tiny default-parser rounding can
    # drastically change ratios of near-zero fitted quadratic coefficients.
    accepted, excluded, x = prepare(pd.read_csv(paths['accepted.csv'], float_precision='round_trip'),
                                   meta['input_lower'], meta['input_upper'])
    if len(excluded) or len(accepted) != meta.get('accepted'):
        raise ValueError('accepted.csv and metadata.json disagree; rerun ELA analysis')
    limits = np.asarray(meta['objective_limits'], dtype=float)
    if limits.shape != (2, 2) or not np.isfinite(limits).all() or np.any(limits[1] < limits[0]):
        raise ValueError('Invalid objective normalization in analysis metadata')
    span = limits[1]-limits[0]
    normalized = (accepted[OUTPUTS].to_numpy()-limits[0])/np.where(span == 0, 1, span)
    ys = {name: normalized[:, j] for j, name in enumerate(OUTPUTS)}
    ys.update({f'weighted_voltage_{w:g}': w*normalized[:, 0]+(1-w)*normalized[:, 1]
               for w in (.25, .5, .75)})
    table = pd.read_csv(paths['features.csv'], float_precision='round_trip')
    if not {'landscape', 'feature', 'value'} <= set(table.columns):
        raise ValueError('Expected landscape,feature,value columns in features.csv')
    if table.duplicated(['landscape', 'feature']).any():
        raise ValueError('Duplicate landscape/feature entries in features.csv')
    full = {}
    for name in landscapes:
        if name not in ys:
            raise ValueError(f'Unknown landscape {name}; choose from {list(ys)}')
        rows = table.loc[table.landscape == name].set_index('feature')
        if set(rows.index) != set(FEATURES):
            raise ValueError(f'{name} does not have the expected 49 features')
        if np.ptp(ys[name]) == 0:
            raise ValueError(f'{name} is constant; cannot generate a nonconstant matched proxy')
        full[name] = pd.to_numeric(rows.value, errors='raise').reindex(FEATURES).to_numpy()
    return accepted, x.to_numpy(), {k: ys[k] for k in landscapes}, full, meta, {
        name: {'path': str(path.resolve()), 'sha256': sha256(path)} for name, path in paths.items()}


def make_subsets(n, repeats=5, fraction=.8, seed=42):
    if repeats < 2 or not 0 < fraction < 1:
        raise ValueError('Use at least 2 repeats and 0 < subset-fraction < 1')
    size = int(np.floor(n*fraction))
    if size < 32:
        raise ValueError(f'Each subset needs at least 32 samples; {n} × {fraction} gives {size}')
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(n, size, replace=False)) for _ in range(repeats)]


def feature_distribution(x, y, subsets, feature_seed=42, ic_sorting='nn'):
    rows = []
    for subset in subsets:
        # Both target and candidate retain the full-pool output normalization.
        values, _ = features(pd.DataFrame(x[subset], columns=INPUTS), y[subset],
                             seed=feature_seed, ic_sorting=ic_sorting)
        rows.append([values[key] if values[key] is not None else np.nan for key in FEATURES])
    return np.asarray(rows, dtype=float)


def select_features(target, threshold=.9):
    """Greedy, deterministic absolute Pearson pruning on target repeats only.

    Finite constants are retained as useful matching constraints. Their Pearson
    correlation is undefined. Candidate failures never change this fixed mask.
    """
    if not 0 < threshold <= 1:
        raise ValueError('correlation-threshold must be in (0, 1]')
    retained, excluded = [], {}
    for j, key in enumerate(FEATURES):
        col = target[:, j]
        if not np.isfinite(col).all():
            excluded[key] = 'undefined or nonfinite in at least one target subset'
            continue
        constant = np.ptp(col) <= 1e-12*max(1., np.max(np.abs(col)))
        redundant = None
        if not constant:
            for k in retained:
                other = target[:, k]
                if np.ptp(other) <= 1e-12*max(1., np.max(np.abs(other))):
                    continue
                correlation = float(np.corrcoef(col, other)[0, 1])
                if np.isfinite(correlation) and abs(correlation) > threshold:
                    redundant = (FEATURES[k], correlation)
                    break
        if redundant:
            excluded[key] = f'correlated with {redundant[0]} (r={redundant[1]:.9g})'
        else:
            retained.append(j)
    if not retained:
        raise ValueError('No finite target features remain')
    return retained, excluded


def distance_scales(target, retained, mode='none'):
    if mode == 'none':
        return np.ones(len(retained))
    if mode != 'target-std':
        raise ValueError('Distance scaling must be none or target-std')
    selected = target[:, retained]
    # Explicit non-paper option; a documented floor avoids division by zero.
    return np.maximum(np.std(selected, axis=0), 1e-6*np.maximum(1, np.abs(selected.mean(axis=0))))


def feature_distance(target, candidate, retained, scales):
    if not np.isfinite(candidate[:, retained]).all() or not np.isfinite(target[:, retained]).all():
        return float('inf'), None
    components = np.asarray([wasserstein_distance(target[:, j], candidate[:, j])/scale
                             for j, scale in zip(retained, scales)])
    return float(components.mean()), components


def validation_diagnostics(target, candidate, retained):
    """Explain unavailable validation scores without silently changing the mask."""
    return {label: {FEATURES[j]: np.flatnonzero(~np.isfinite(matrix[:, j])).tolist()
                    for j in retained if not np.isfinite(matrix[:, j]).all()}
            for label, matrix in [('undefined_target_features', target),
                                  ('undefined_proxy_features', candidate)]}


def terminal(rng):
    choice = int(rng.integers(4))
    if choice < 2:
        return Node(('x', 'index')[choice])
    # Fixed real constants and ephemeral random reals; rand is sampled once.
    value = float(rng.choice([-2, -1, .5, 1, 2])) if choice == 2 else float(rng.uniform(-5, 5))
    return Node('constant', value=value)


def random_tree(rng, min_depth, max_depth, max_nodes, method=None):
    for _ in range(200):
        depth = int(rng.integers(min_depth, max_depth+1))
        full = (rng.random() < .5) if method is None else method == 'full'
        def grow(level):
            if level == depth or (not full and level >= min_depth and rng.random() < .3):
                return terminal(rng)
            op = rng.choice(list(ARITY))
            return Node(str(op), tuple(grow(level+1) for _ in range(ARITY[op])))
        tree = grow(0)
        if tree.size <= max_nodes:
            return tree
    raise ValueError('Unable to initialize a tree: increase max-nodes or reduce tree depths')


def paths(tree, prefix=()):
    result = [prefix]
    for j, child in enumerate(tree.children):
        result.extend(paths(child, prefix+(j,)))
    return result


def subtree(tree, path):
    for index in path:
        tree = tree.children[index]
    return tree


def replace(tree, path, new):
    if not path:
        return new
    children = list(tree.children)
    children[path[0]] = replace(children[path[0]], path[1:], new)
    return Node(tree.op, tuple(children), tree.value)


def crossover(a, b, rng, max_depth, max_nodes):
    pa, pb = paths(a), paths(b)
    ia, ib = pa[int(rng.integers(len(pa)))], pb[int(rng.integers(len(pb)))]
    first, second = replace(a, ia, subtree(b, ib)), replace(b, ib, subtree(a, ia))
    return tuple(child if child.depth <= max_depth and child.size <= max_nodes else parent
                 for child, parent in ((first, a), (second, b)))


def mutate(tree, rng, max_depth, max_nodes):
    options = paths(tree)
    path = options[int(rng.integers(len(options)))]
    new = random_tree(rng, 0, min(3, max_depth-len(path)), max_nodes)
    child = replace(tree, path, new)
    return child if child.depth <= max_depth and child.size <= max_nodes else tree


@dataclass
class Candidate:
    tree: Node
    fitness: float
    offset: float = 0.
    scale: float = 1.
    distribution: np.ndarray = None
    components: np.ndarray = None


class LandscapeScorer:
    def __init__(self, x, subsets, target, retained, scales, feature_seed, ic_sorting, seed):
        self.x, self.subsets, self.target = x, subsets, target
        self.retained, self.scales = retained, scales
        self.feature_seed, self.ic_sorting = feature_seed, ic_sorting
        self.cache, self.value_cache = {}, {}
        self.evaluations = 0
        # Finite domain probes supplement the paper's sample-pool checks.
        self.probes = np.vstack([qmc.Sobol(3, scramble=True, seed=seed).random_base2(8),
                                 list(itertools.product((0., 1.), repeat=3)), [.5, .5, .5]])

    def __call__(self, tree):
        if tree in self.cache:
            return self.cache[tree]
        raw = evaluate_tree(tree, self.x)
        bad = Candidate(tree, float('inf'))
        if not np.isfinite(raw).all():
            self.cache[tree] = bad
            return bad
        with np.errstate(all='ignore'):
            offset, scale = float(np.min(raw)), float(np.ptp(raw))
            y = (raw-offset)/scale
            probe = (evaluate_tree(tree, self.probes)-offset)/scale
        if scale <= 0 or not np.isfinite(scale) or not np.isfinite(y).all() or not np.isfinite(probe).all():
            self.cache[tree] = bad
            return bad
        signature = hashlib.sha256(y.tobytes()).hexdigest()
        if signature not in self.value_cache:
            self.evaluations += 1
            distribution = feature_distribution(self.x, y, self.subsets, self.feature_seed, self.ic_sorting)
            fit, components = feature_distance(self.target, distribution, self.retained, self.scales)
            self.value_cache[signature] = (fit, distribution, components)
        fit, distribution, components = self.value_cache[signature]
        result = Candidate(tree, fit, offset, scale, distribution, components)
        self.cache[tree] = result
        return result


def artifact(candidate, landscape, meta, seed):
    return {'schema_version': 1, 'kind': 'ela_matched_scalar_benchmark', 'paper': PAPER,
            'landscape': landscape, 'tree': candidate.tree.to_dict(),
            'expression': candidate.tree.expression(), 'vector_root_reduction': 'sum',
            'input_columns': INPUTS, 'input_lower': meta['input_lower'],
            'input_upper': meta['input_upper'], 'raw_offset': candidate.offset,
            'raw_scale': candidate.scale, 'feature_distance': candidate.fitness,
            'seed': seed, 'output': 'dimensionless, normalized on the training design pool',
            'warning': 'A synthetic optimization benchmark, not a physical response surrogate; '
                       'no guarantee of matching values, minimizers, or multiobjective trade-offs.'}


def evolve(scorer, args, progress):
    rng = np.random.default_rng(args.seed)
    population = [random_tree(rng, args.min_depth, args.max_depth, args.max_nodes,
                              method='full' if j % 2 else 'grow') for j in range(args.population)]
    archive = {}
    started = time.monotonic()
    for generation in range(args.generations+1):
        ranked = [scorer(tree) for tree in population]
        for candidate in ranked:
            if np.isfinite(candidate.fitness):
                archive[candidate.tree] = candidate
        best = sorted(archive.values(), key=lambda c: (c.fitness, c.tree.size, c.tree.expression()))
        progress(generation, ranked, best, time.monotonic()-started)
        if generation == args.generations:
            break
        def tournament():
            contestants = rng.integers(len(ranked), size=args.tournament_size)
            return min((ranked[j] for j in contestants), key=lambda c: c.fitness).tree
        offspring = [tournament() for _ in population]
        for j in range(0, len(offspring)-1, 2):
            if rng.random() < args.crossover:
                offspring[j], offspring[j+1] = crossover(offspring[j], offspring[j+1], rng,
                                                        args.max_depth, args.max_nodes)
        for j in range(len(offspring)):
            if rng.random() < args.mutation:
                offspring[j] = mutate(offspring[j], rng, args.max_depth, args.max_nodes)
        population = offspring
    if not best:
        raise ValueError('No finite proxies found. Increase population/generations or try another seed. '
                         'Search diagnostics are saved in history.csv.')
    return best[:args.top_k]


def export_proxy(destination, candidate, name, meta, seed):
    destination.mkdir()
    model = artifact(candidate, name, meta, seed)
    save_json(destination/'proxy.json', model)
    # A portable folder: only this small runtime plus NumPy is needed later.
    shutil.copyfile(Path(__file__).with_name('proxy_functions.py'), destination/'proxy_runtime.py')
    (destination/'proxy.py').write_text(
        '"""Synthetic ELA benchmark. objective(x) accepts physical AEMEC inputs."""\n'
        'from pathlib import Path\nimport importlib.util\nimport sys\n\n'
        '_root = Path(__file__).resolve().parent\n'
        '_spec = importlib.util.spec_from_file_location("_ela_proxy_runtime", _root / "proxy_runtime.py")\n'
        '_runtime = importlib.util.module_from_spec(_spec)\n'
        'sys.modules[_spec.name] = _runtime\n_spec.loader.exec_module(_runtime)\n'
        'objective = _runtime.ProxyFunction(_root / "proxy.json")\n')
    return model


def run(args):
    if args.population < 4 or args.generations < 0 or args.top_k < 1 or args.tournament_size < 2:
        raise ValueError('Require population >= 4, generations >= 0, top-k >= 1, tournament-size >= 2')
    if not 0 <= args.crossover <= 1 or not 0 <= args.mutation <= 1:
        raise ValueError('Crossover and mutation probabilities must be in [0, 1]')
    if not 0 <= args.min_depth <= args.max_depth <= 12 or not 2 <= args.max_nodes <= 4096:
        raise ValueError('Require 0 <= min-depth <= max-depth <= 12 and 2 <= max-nodes <= 4096')
    if len(set(args.landscapes)) != len(args.landscapes):
        raise ValueError('Landscape names must be unique')
    accepted, x, ys, full, meta, sources = load_analysis(args.analysis, args.landscapes)
    subsets = make_subsets(len(x), args.repeats, args.subset_fraction, args.seed)
    validation = make_subsets(len(x), args.repeats, args.subset_fraction, args.seed+1000003)
    destination = Path(args.output)
    if destination.exists():
        raise ValueError(f'Output already exists: {destination}; choose a new directory')
    destination.mkdir(parents=True)
    feature_seed = int(meta['seed'])
    ic_sorting = meta['feature_settings']['ic_sorting']
    manifest = {'schema_version': 1, 'paper': PAPER, 'sources': sources, 'status': 'running',
                'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'sample_count': len(x), 'paper_sample_recommendation': 150*x.shape[1],
                'feature_seed': feature_seed, 'ic_sorting': ic_sorting,
                'subset_indices': [s.tolist() for s in subsets],
                'validation_subset_indices': [s.tolist() for s in validation],
                'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'pflacco']},
                'distance': 'mean of per-feature univariate Wasserstein distances between subset estimates',
                'validation': 'new random subsets of the SAME evaluated sample pool, not held-out CFD',
                'objective_normalization': meta['objective_limits'], 'completed_landscapes': []}
    save_json(destination/'manifest.json', manifest)
    summaries = []
    try:
        for name, y in ys.items():
            out = destination/name
            out.mkdir()
            print(f'{name}: preparing {args.repeats} target ELA subsets ({len(subsets[0])}/{len(x)} designs each)', flush=True)
            # Ensure the supplied feature vector is actually from these observations
            # under the current feature settings/software, rather than silently mixing runs.
            actual = feature_distribution(x, y, [np.arange(len(x))], feature_seed, ic_sorting)[0]
            if not np.allclose(actual, full[name], rtol=1e-6, atol=1e-9, equal_nan=True):
                raise ValueError(f'{name}: features.csv does not match accepted.csv/settings/runtime; '
                                 'regenerate ELA analysis in this environment')
            target = feature_distribution(x, y, subsets, feature_seed, ic_sorting)
            retained, excluded = select_features(target, args.correlation_threshold)
            scales = distance_scales(target, retained, args.distance_scaling)
            pd.DataFrame(target, columns=FEATURES).to_csv(out/'target_feature_distributions.csv', index_label='repeat')
            save_json(out/'feature_selection.json', {'retained': [FEATURES[j] for j in retained],
                                                    'excluded': excluded, 'scales': scales.tolist()})
            scorer = LandscapeScorer(x, subsets, target, retained, scales, feature_seed, ic_sorting, args.seed)
            history = []
            def progress(generation, ranked, best, elapsed):
                finite = [c.fitness for c in ranked if np.isfinite(c.fitness)]
                history.append({'generation': generation, 'best_distance': best[0].fitness if best else None,
                                'population_median': float(np.median(finite)) if finite else None,
                                'finite_population': len(finite), 'distinct_expressions': len(scorer.cache),
                                'ela_candidate_evaluations': scorer.evaluations, 'elapsed_seconds': elapsed})
                pd.DataFrame(history).to_csv(out/'history.csv', index=False)
                if best:
                    save_json(out/'best_so_far.json', artifact(best[0], name, meta, args.seed))
                score = f'{best[0].fitness:.6g}' if best else 'none'
                print(f'{name}: generation {generation}/{args.generations}, best={score}, '
                      f'finite={len(finite)}/{len(ranked)}, elapsed={elapsed:.1f}s', flush=True)
            best = evolve(scorer, args, progress)
            target_validation = feature_distribution(x, y, validation, feature_seed, ic_sorting)
            pd.DataFrame(target_validation, columns=FEATURES).to_csv(out/'validation_target_distributions.csv', index_label='repeat')
            for rank, candidate in enumerate(best, 1):
                folder = out/f'proxy_{rank:02d}'
                model = export_proxy(folder, candidate, name, meta, args.seed)
                generated = ProxyFunction(model)(accepted[INPUTS].to_numpy())
                candidate_full = feature_distribution(x, generated, [np.arange(len(x))], feature_seed, ic_sorting)[0]
                candidate_validation = feature_distribution(x, generated, validation, feature_seed, ic_sorting)
                validation_score, _ = feature_distance(target_validation, candidate_validation, retained, scales)
                save_json(folder/'validation_diagnostics.json',
                          validation_diagnostics(target_validation, candidate_validation, retained))
                pd.DataFrame(candidate.distribution, columns=FEATURES).to_csv(folder/'feature_distributions.csv', index_label='repeat')
                pd.DataFrame(candidate_validation, columns=FEATURES).to_csv(folder/'validation_feature_distributions.csv', index_label='repeat')
                comparison = pd.DataFrame({'feature': FEATURES, 'target_full': full[name], 'proxy_full': candidate_full,
                                           'target_subset_mean': np.mean(target, axis=0),
                                           'proxy_subset_mean': np.mean(candidate.distribution, axis=0),
                                           'retained': [j in retained for j in range(len(FEATURES))]})
                distances = dict(zip(retained, candidate.components))
                comparison['wasserstein_component'] = [distances.get(j, np.nan) for j in range(len(FEATURES))]
                comparison.to_csv(folder/'feature_comparison.csv', index=False)
                samples = accepted[INPUTS].copy()
                samples['target_normalized'] = y
                samples['proxy_normalized'] = generated
                samples.to_csv(folder/'sample_values.csv', index=False)
                summaries.append({'landscape': name, 'rank': rank, 'training_distance': candidate.fitness,
                                  'resampling_validation_distance': validation_score if np.isfinite(validation_score) else None,
                                  'validation_valid': bool(np.isfinite(validation_score)),
                                  'retained_features': len(retained), 'tree_nodes': candidate.tree.size,
                                  'tree_depth': candidate.tree.depth, 'expression': candidate.tree.expression(),
                                  'artifact': str((folder/'proxy.json').relative_to(destination))})
            manifest['completed_landscapes'].append(name)
            pd.DataFrame(summaries).to_csv(destination/'summary.csv', index=False)
            save_json(destination/'manifest.json', manifest)
        manifest['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as exc:
        manifest['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        manifest['error'] = str(exc)
        save_json(destination/'manifest.json', manifest)
        raise
    save_json(destination/'manifest.json', manifest)
    (destination/'report.md').write_text(
        '# ELA-matched synthetic proxy functions\n\n'
        f'Generated {len(summaries)} scalar proxies from {len(x)} existing evaluated designs. '
        f'The paper uses 150 × dimension = {150*x.shape[1]} designs; no new CFD was run.\n\n'
        'See summary.csv for distances and feature_comparison.csv for all 49 features. '
        'Smaller distance is better, but there is no universal acceptance threshold.\n\n'
        'The search minimizes mean per-feature Wasserstein distance on a fixed target-derived '
        'feature subset. Raw feature units are used unless target-std scaling was explicitly requested. '
        'Features are never pooled as interchangeable observations.\n\n'
        'Validation uses different random subsets of the same pool, not independent CFD observations. '
        'Unavailable scores are explained by validation_diagnostics.json in each proxy folder. '
        'Five resamples provide a noisy correlation estimate; inspect excluded features and repeat seeds.\n\n'
        'Each function is an independent, dimensionless optimization benchmark. Matching ELA features '
        'does not guarantee agreement of response values, optima, monotonicity, or the joint Pareto front. '
        'The finite domain probes cannot prove numerical validity everywhere.\n\n'
        f'Method and documented implementation choices: {PAPER}; ELA/PROXIES.md.\n')
    return summaries
