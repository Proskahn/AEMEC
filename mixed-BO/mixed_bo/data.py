"""Human-defined material catalog and explicit mixed-design data contracts."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

CONTINUOUS = ['membrane_thickness_um', 'water_inlet_temperature_k', 'ptl_porosity']
CATEGORICAL = ['anode_pte', 'membrane_type']
OBJECTIVES = ['cell_voltage_v', 'crossover_rate_mol_s']
DESIGN = CONTINUOUS + CATEGORICAL
LOWER = np.array([10., 298.15, .4])
UPPER = np.array([100., 353.15, .8])


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_catalog(path, allow_demo=False, require_descriptors=True):
    catalog = json.loads(Path(path).read_text())
    if catalog.get('schema') != 1 or set(catalog.get('variables', {})) != set(CATEGORICAL):
        raise ValueError('Catalog requires schema=1 and variables anode_pte and membrane_type')
    if catalog.get('synthetic_demo', False) and not allow_demo:
        raise ValueError('Synthetic descriptor catalog: pass --allow-demo only for a demonstration')
    for name in CATEGORICAL:
        block = catalog['variables'][name]
        types = block.get('types', [])
        if len(types) != 3 or any(not isinstance(t, str) or not t.strip() for t in types) or len(set(types)) != 3:
            raise ValueError(f'{name}: define exactly three unique, nonempty type names')
        if not require_descriptors:
            continue
        if set(block.get('descriptors', {})) != set(types):
            raise ValueError(f'{name}: descriptors must cover exactly the three types')
        for family in ['geometry', 'physics']:
            schema = block.get(family, [])
            if not schema or any(not isinstance(s, dict) or not s.get('name') or not s.get('unit') for s in schema):
                raise ValueError(f'{name}.{family}: define descriptor names and units')
            if len({s['name'] for s in schema}) != len(schema):
                raise ValueError(f'{name}.{family}: duplicate descriptor names')
            for category in types:
                raw = block['descriptors'][category].get(family)
                if (not isinstance(raw, list) or len(raw) != len(schema) or
                        any(v is None or isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw)):
                    raise ValueError(f'{name}/{category}/{family}: fill all human-defined numeric descriptor values')
                if not np.isfinite(raw).all():
                    raise ValueError(f'{name}/{category}/{family}: descriptor values must be finite')
    return catalog


def check_design(frame, catalog):
    missing = set(DESIGN)-set(frame.columns)
    if missing:
        raise ValueError(f'Missing mixed-design columns {sorted(missing)}. Existing ELA rows need their TRUE material '
                         'labels; do not assign random categories or duplicate outputs across categories.')
    result = frame.copy().reset_index(drop=True)
    x = result[CONTINUOUS].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    if not np.isfinite(x).all() or np.any(x < LOWER) or np.any(x > UPPER):
        raise ValueError('Continuous inputs must be finite and within the existing AEMEC study bounds')
    result[CONTINUOUS] = x
    for name in CATEGORICAL:
        if not result[name].isin(catalog['variables'][name]['types']).all():
            raise ValueError(f'Unknown or missing {name}; use the exact type names from the descriptor catalog')
    return result


def read_evaluations(path, catalog):
    frame = pd.read_csv(path)
    missing = set(DESIGN+OBJECTIVES)-set(frame.columns)
    if missing:
        raise ValueError(f'Evaluations are missing columns {sorted(missing)}; material labels must identify '
                         'the actual evaluated materials. Unlabeled ELA data cannot identify material effects.')
    reasons = pd.Series('', index=frame.index)
    if 'status' in frame:
        valid = frame.status.astype(str).str.strip().str.lower().isin(['complete', 'completed', 'ok', 'success'])
        reasons.loc[~valid] = 'unsuccessful_status'
    y = frame[OBJECTIVES].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    bad = ~np.isfinite(y).all(axis=1) | (y[:, 0] <= 0) | (y[:, 1] < 0)
    reasons.loc[(reasons == '') & bad] = 'missing_or_invalid_objectives'
    excluded = frame.loc[reasons != ''].copy()
    excluded['exclusion_reason'] = reasons[reasons != '']
    accepted = check_design(frame.loc[reasons == ''], catalog)
    accepted[OBJECTIVES] = y[reasons.to_numpy() == '']
    if len(accepted) < 12:
        raise ValueError('At least 12 valid, category-labeled observations are required for GP training and CV')
    if accepted[DESIGN].duplicated().any():
        raise ValueError('Duplicate complete mixed designs: aggregate replicates before training to avoid CV leakage')
    if 'sample_id' in accepted and (accepted.sample_id.isna().any() or accepted.sample_id.duplicated().any()):
        raise ValueError('sample_id must be nonempty and unique')
    for key in ['ptl_side', 'target_current_density_a_m2']:
        if key in accepted and (accepted[key].isna().any() or accepted[key].nunique() != 1):
            raise ValueError(f'Evaluations mix or omit operating setting {key}')
    if np.any(np.ptp(accepted[OBJECTIVES].to_numpy(), axis=0) == 0):
        raise ValueError('Both objective columns must vary')
    return accepted, excluded


def initial_design(catalog, per_combination=4, seed=42):
    if per_combination < 1:
        raise ValueError('--per-combination must be positive')
    combinations = list(itertools.product(*(catalog['variables'][k]['types'] for k in CATEGORICAL)))
    blocks = []
    for index, combo in enumerate(combinations):
        # Scrambled Sobol prefix; arbitrary counts are permitted without pretending
        # that a non-power-of-two prefix has the balance of a complete Sobol net.
        unit = qmc.Sobol(3, scramble=True, seed=seed+index).random_base2(
            int(np.ceil(np.log2(per_combination))))[:per_combination]
        block = pd.DataFrame(qmc.scale(unit, LOWER, UPPER), columns=CONTINUOUS)
        for name, value in zip(CATEGORICAL, combo):
            block[name] = value
        blocks.append(block)
    result = pd.concat(blocks, ignore_index=True)
    result.insert(0, 'sample_id', np.arange(len(result)))
    for name in OBJECTIVES:
        result[name] = np.nan
    result['status'] = 'pending'
    return result
