"""Load and identify the pinned OptunaHub multiobjective CMA sampler.

The optimizer itself lives in the unmodified, MIT-licensed upstream module.
The common OptunaAlgorithm adapter handles ask/tell and shared initialization.
No sampler download, DEAP dependency, or synthetic objective is needed at runtime.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import optuna


OPTUNA_VERSION = '4.9.0'
UPSTREAM_COMMIT = '7508d98688d3eb55aab09002ea6a8dedfbe5b2c6'
UPSTREAM_URL = ('https://github.com/optuna/optunahub-registry/blob/'
                f'{UPSTREAM_COMMIT}/package/samplers/mocma/mocma.py')
SOURCE_PATH = Path(__file__).with_name('_vendor')/'mocma'/'mocma.py'


def require_mocma():
    """Fail before CFD if the sampler or its tested Optuna runtime is unavailable.

    Upstream imports private Optuna APIs, so an arbitrary upgrade is not safe.
    Pinning both pieces makes the benchmark reproducible and works offline on
    the server once the standard Python dependencies have been installed.
    """
    if optuna.__version__ != OPTUNA_VERSION:
        raise ValueError(f'MoCmaSampler requires the tested Optuna {OPTUNA_VERSION}; '
                         'install with: python -m pip install -r benchmark/requirements.txt')
    try:
        from ._vendor.mocma import MoCmaSampler
    except ImportError as exc:
        raise ValueError('Cannot load the bundled OptunaHub MoCmaSampler. Copy the complete '
                         'benchmark folder and install benchmark/requirements.txt') from exc
    return MoCmaSampler


def mocma_protocol(population_size):
    """Record the actual upstream implementation and its experimental protocol."""
    return {
        'implementation': 'OptunaHub MoCmaSampler (unmodified upstream)',
        'upstream_url': UPSTREAM_URL,
        'upstream_commit': UPSTREAM_COMMIT,
        'source_sha256': hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        'population_size': population_size,
        'initial_sigma': 1/6,
        'parents': 'shared successful designs; charged random fills until population_size successes',
        'updates': 'generation-based; selection/adaptation at next ask after a complete generation',
        'boundary_handling': 'clip to normalized input box',
        'selection': 'upstream Pareto-rank/hypervolume code with a population-derived internal reference',
        'reporting_reference': 'fixed common benchmark reference; separate from internal selection',
        'failures': 'Optuna FAIL; excluded from population; later charged attempts fill missing slots',
        'known_upstream_caveats': [
            'Covariance uses a vector dot product where a rank-one outer product is expected.',
            'Overflow-front selection retains the smallest hypervolume contributor first.',
        ],
    }
