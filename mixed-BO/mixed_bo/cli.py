"""Explicit file-based train / propose / evaluate / append / refit workflow."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .data import CATEGORICAL, CONTINUOUS, OBJECTIVES, LOWER, UPPER, read_catalog, read_evaluations, initial_design
from .representation import fit_representation, load_representation
from .surrogate import train_surrogates
from .acquisition import suggest

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def new_bundle(destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f'Output already exists: {destination}; choose a fresh output directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.mixed-bo-', dir=destination.parent) as name:
        work = Path(name)
        yield work
        versions = {k: importlib.metadata.version(k) for k in
                    ['numpy', 'pandas', 'scipy', 'torch', 'scikit-learn', 'joblib']}
        (work/'environment.json').write_text(json.dumps(versions, indent=2)+'\n')
        work.rename(destination)
    print(f'Saved: {destination.resolve()}', flush=True)


def write_csv_new(frame, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing results, including source CSVs.
    with destination.open('x') as handle:
        frame.to_csv(handle, index=False)
    print(f'Saved {len(frame)} rows: {destination.resolve()}', flush=True)


def synthetic_evaluations(frame, catalog):
    """Software demo only: these are deliberately artificial objective functions."""
    result = frame.copy()
    x = (result[CONTINUOUS].to_numpy()-LOWER)/(UPPER-LOWER)
    a, m = [np.array([catalog['variables'][k]['types'].index(t) for t in frame[k]], float) for k in CATEGORICAL]
    result[OBJECTIVES[0]] = 1.7+.3*x[:, 0]-.08*x[:, 1]+.02*x[:, 2]+.035*a+.025*m+.03*x[:, 1]*a
    result[OBJECTIVES[1]] = 1e-9*(1+1/(.3+x[:, 0])+.1*x[:, 2]+.15*a-.15*m+.05*x[:, 1]*m)
    result['status'] = 'complete'
    result['synthetic_demo'] = True
    return result


def synthetic_auxiliaries(frame, catalog):
    """Cheap demo-only functions of x; never evaluate or read expensive y."""
    from .data import DESIGN
    result = frame[DESIGN].copy()
    x = (result[CONTINUOUS].to_numpy()-LOWER)/(UPPER-LOWER)
    a, m = [np.array([catalog['variables'][k]['types'].index(t) for t in frame[k]], float) for k in CATEGORICAL]
    result['g__demo_path_length'] = x[:, 0]*(1+.15*a)
    result['g__demo_pore_shape'] = x[:, 2]+.2*a+.1*m
    result['p__demo_resistance_proxy'] = (.2+x[:, 0])/(1+.4*x[:, 1]+.2*m)
    result['p__demo_transport_proxy'] = 1/(.3+x[:, 0])+.1*a+.1*x[:, 2]-.1*m
    return result


def add_vae_options(parser):
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--latent-dim', type=int, default=4)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--beta', type=float, default=.01)
    parser.add_argument('--geometry-weight', type=float, default=1.)
    parser.add_argument('--physics-weight', type=float, default=1.)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--batch-size', type=int, default=128)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Descriptor-informed VAE + GP mixed-variable Bayesian optimization')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['fit-vae', 'sample', 'train', 'suggest', 'predict', 'append', 'demo']:
        p = commands.add_parser(name)
        p.add_argument('--output', type=Path, required=True)
        if name not in ['predict', 'append']:
            p.add_argument('--seed', type=int, default=42)
        if name in ['fit-vae', 'sample', 'append']:
            p.add_argument('--catalog', type=Path, required=True)
        if name != 'demo':
            p.add_argument('--allow-demo', action='store_true', help='Explicitly permit a synthetic descriptor catalog')
        if name in ['fit-vae', 'demo']:
            add_vae_options(p)
        if name == 'fit-vae':
            p.add_argument('--pretrain-csv', type=Path,
                           help='Cheap full designs with g__... and p__... columns; expensive y is never used')
            p.add_argument('--per-combination', type=int, default=64,
                           help='Generated cheap designs per material pair when --pretrain-csv is omitted')
        if name == 'demo':
            p.add_argument('--pretrain-per-combination', type=int, default=64,
                           help='Cheap pretraining sample count per material pair, separate from GP observations')
        if name in ['sample', 'demo']:
            p.add_argument('--per-combination', type=int, default=4)
        if name in ['train', 'append']:
            p.add_argument('csv', type=Path)
        if name == 'train':
            p.add_argument('--representation', type=Path, required=True)
            p.add_argument('--encoding', choices=['vae', 'descriptors', 'onehot'], default='vae')
            p.add_argument('--cv-mode', choices=['random', 'combination', *CATEGORICAL], default='random')
        if name in ['train', 'demo']:
            p.add_argument('--folds', type=int, default=5)
            p.add_argument('--restarts', type=int, default=2)
        if name in ['suggest', 'predict']:
            p.add_argument('--model-dir', type=Path, required=True)
        if name in ['suggest', 'demo']:
            p.add_argument('--candidates-per-combination', type=int, default=512)
        if name == 'suggest':
            p.add_argument('--reference-point', type=float, nargs=2, help='Physical-unit voltage and crossover; both minimized')
            p.add_argument('--pending', type=Path, help='Already proposed designs to exclude')
        if name == 'predict':
            p.add_argument('csv', type=Path)
        if name == 'append':
            p.add_argument('--new-evaluations', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if hasattr(args, 'seed') and not 0 <= args.seed <= 2**32-10000:
            raise ValueError('--seed must be nonnegative and below 2**32-10000')
        if args.command in ['fit-vae', 'sample', 'append']:
            catalog = read_catalog(args.catalog, args.allow_demo, require_descriptors=(
                args.command == 'fit-vae' and args.pretrain_csv is None))
        if args.command in ['fit-vae', 'demo']:
            vae_options = dict(epochs=args.epochs, seed=args.seed, latent_dim=args.latent_dim,
                               hidden_dim=args.hidden_dim, beta=args.beta, geometry_weight=args.geometry_weight,
                               physics_weight=args.physics_weight, lr=args.learning_rate, batch_size=args.batch_size)
        if args.command == 'fit-vae':
            with new_bundle(args.output) as work:
                designs = pd.read_csv(args.pretrain_csv) if args.pretrain_csv else None
                fit_representation(catalog, work, designs=designs, per_combination=args.per_combination, **vae_options)
        elif args.command == 'sample':
            write_csv_new(initial_design(catalog, args.per_combination, args.seed), args.output)
        elif args.command == 'train':
            representation = load_representation(args.representation)
            if representation['catalog'].get('synthetic_demo') and not args.allow_demo:
                raise ValueError('Synthetic representation: pass --allow-demo only for demonstration data')
            with new_bundle(args.output) as work:
                train_surrogates(args.csv, representation, work, args.encoding, args.folds, args.cv_mode,
                                 args.seed, args.restarts)
                (work/'source.json').write_text(json.dumps({'path': str(args.csv.resolve()),
                    'sha256': hashlib.sha256(args.csv.read_bytes()).hexdigest()}, indent=2)+'\n')
        elif args.command in ['suggest', 'predict']:
            from .surrogate import load_surrogates, predict
            models, representation, metadata = load_surrogates(args.model_dir)
            if representation['catalog'].get('synthetic_demo') and not args.allow_demo:
                raise ValueError('Synthetic model: pass --allow-demo only for demonstration predictions')
            if args.command == 'suggest':
                with new_bundle(args.output) as work:
                    suggest(args.model_dir, work, args.seed, args.candidates_per_combination,
                            args.reference_point, args.pending)
            else:
                frame = pd.read_csv(args.csv)
                means, stds = predict(models, frame, representation, metadata['encoding'])
                for j, objective in enumerate(OBJECTIVES):
                    frame[f'{objective}_gp_mean'] = means[:, j]
                    frame[f'{objective}_gp_std'] = stds[:, j]
                write_csv_new(frame, args.output)
        elif args.command == 'append':
            combined = pd.concat([pd.read_csv(args.csv), pd.read_csv(args.new_evaluations)], ignore_index=True)
            # Validate the combined accepted pool before publishing a new source file.
            with tempfile.TemporaryDirectory() as directory:
                check = Path(directory)/'combined.csv'
                combined.to_csv(check, index=False)
                read_evaluations(check, catalog)
            write_csv_new(combined, args.output)
        elif args.command == 'demo':
            catalog = read_catalog(ROOT/'examples/materials.demo.json', allow_demo=True)
            with new_bundle(args.output) as work:
                cheap = synthetic_auxiliaries(initial_design(catalog, args.pretrain_per_combination, args.seed+101), catalog)
                cheap.to_csv(work/'cheap_pretraining.csv', index=False)
                representation = fit_representation(catalog, work/'vae', designs=cheap, **vae_options)
                synthetic_evaluations(initial_design(catalog, args.per_combination, args.seed), catalog).to_csv(
                    work/'synthetic_evaluations.csv', index=False)
                model_dir = work/'gp'
                model_dir.mkdir()
                train_surrogates(work/'synthetic_evaluations.csv', representation, model_dir,
                                 folds=args.folds, seed=args.seed, restarts=args.restarts)
                suggestion_dir = work/'suggestion'
                suggestion_dir.mkdir()
                suggest(model_dir, suggestion_dir, args.seed, args.candidates_per_combination)
                # Resolve the published path rather than leaving a temporary path in provenance.
                info_path = suggestion_dir/'suggestion.json'
                info = json.loads(info_path.read_text())
                info['source_model_directory'] = str((args.output/'gp').resolve())
                info_path.write_text(json.dumps(info, indent=2)+'\n')
                (work/'DEMO_ONLY.txt').write_text('Synthetic descriptors and objectives. No CFD or real material training.\n')
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f'Mixed-BO error: {exc}\n')


if __name__ == '__main__':
    main()
