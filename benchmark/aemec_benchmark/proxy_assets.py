"""Load immutable ELA proxy assets used only inside the proposal algorithm."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import shlex
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_proxy_paths(args):
    """Find one voltage/crossover pair without assuming a generated folder name.

    Explicit paths are interpreted relative to the calling shell directory and
    support '~'. Discovery is restricted to this checkout's ELA/results. Pair
    only siblings of the same rank in one generation run: combining unrelated
    runs or silently choosing the newest result would change the benchmark.
    Artifact content, objective roles and input domains are validated separately
    when the algorithm assets are loaded.
    """
    if args.proxy_rank < 1:
        raise ValueError('proxy-rank must be positive')
    if args.proxy_files is not None:
        paths = [path.expanduser().resolve() for path in args.proxy_files]
    else:
        def pair(root):
            return [root/name/f'proxy_{args.proxy_rank:02d}'/'proxy.json'
                    for name in ('cell_voltage_v', 'crossover_rate_mol_s')]

        if args.proxy_root is not None:
            # An explicit but incorrect root is an error, not permission to
            # substitute a different proxy function found somewhere else.
            paths = pair(args.proxy_root.expanduser().resolve())
        else:
            search_root = ROOT/'ELA/results'
            pattern = f'**/cell_voltage_v/proxy_{args.proxy_rank:02d}/proxy.json'
            roots = sorted({path.parents[2].resolve() for path in search_root.glob(pattern)
                            if all(item.is_file() for item in pair(path.parents[2]))})
            if not roots:
                raise ValueError(
                    f'No complete rank-{args.proxy_rank} proxy pair found under {search_root}. '
                    'Pass --proxy-root /path/to/generated_run (containing cell_voltage_v/ and '
                    'crossover_rate_mol_s/), or --proxy-files /path/to/voltage.json /path/to/crossover.json. '
                    'If no proxy JSON files exist on this machine, copy the generated proxies '
                    'or generate them with ELA/generate_proxies.py fit; evaluated CSVs are not proxy functions.')
            if len(roots) > 1:
                options = '\n'.join(f'  --proxy-root {shlex.quote(str(root))}' for root in roots)
                raise ValueError(f'Multiple rank-{args.proxy_rank} proxy pairs found. Choose one explicitly:\n{options}')
            paths = pair(roots[0])
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError('Proxy JSON file(s) not found:\n  '+'\n  '.join(missing)+
                         '\nSet --proxy-root to the parent of the two objective folders, '
                         'or pass both JSON paths with --proxy-files. '
                         'The generated files must exist on this machine.')
    return paths


@dataclass(frozen=True)
class ProxyAssets:
    """Frozen expression models and their provenance, never CFD observations."""
    models: tuple
    lower: np.ndarray
    upper: np.ndarray
    metadata: dict


def load_proxy_assets(paths, lower, upper):
    """Require the proposal models to cover exactly the benchmark input domain.

    Proxy outputs stay dimensionless. Their scaling never replaces the fixed
    CFD objective scaling or hypervolume reference. Existing training samples
    are not loaded into a benchmark run's observation history.
    """
    from ELA.proxy_functions import ProxyFunction
    from .runner import INPUTS

    paths = tuple(Path(path).expanduser().resolve() for path in paths)
    if len(paths) != 2:
        raise ValueError('ELA proxy search requires voltage and crossover proxy files, in that order')
    models = tuple(ProxyFunction(path) for path in paths)
    for model, name in zip(models, ('cell_voltage_v', 'crossover_rate_mol_s')):
        if model.input_columns != INPUTS or model.landscape != name:
            raise ValueError('Proxy objective roles/input columns do not match the benchmark')
        if not np.array_equal(model.lower, lower) or not np.array_equal(model.upper, upper):
            raise ValueError('Proxy domains must match the CFD study bounds exactly')
    provenance = []
    # Standard generator artifacts have a parent run manifest. Record its
    # historical data cost when known; do not silently count it as zero.
    for manifest in sorted({path.parents[2]/'manifest.json' for path in paths}):
        if manifest.is_file():
            data = json.loads(manifest.read_text())
            provenance.append({'path': str(manifest), 'sha256': file_hash(manifest),
                               'historical_sample_count': data.get('sample_count'),
                               'sources': data.get('sources')})
    metadata = {
        'models': [{'path': str(path), 'sha256': file_hash(path)} for path in paths],
        'training_provenance': provenance,
        'prior_information': 'Frozen ELA expressions derived from historical CFD data. Historical '
                             'sampling and fitting costs are additional to the online CFD attempt budget; '
                             'missing provenance means unknown cost, not zero.',
        'role': 'Proposal generation only; model values never enter CFD observation tables or metrics.',
    }
    return ProxyAssets(models, np.asarray(lower, dtype=float).copy(),
                       np.asarray(upper, dtype=float).copy(), metadata)

