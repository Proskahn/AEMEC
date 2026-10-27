"""Generate cheap mathematical benchmark functions by matching AEMEC ELA features."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit', help='Evolve proxies from an existing ELA analysis directory')
    fit.add_argument('analysis', type=Path, help='Directory with features.csv, accepted.csv, metadata.json')
    fit.add_argument('--output', type=Path, required=True, help='New output directory (never overwritten)')
    fit.add_argument('--landscapes', nargs='+', default=['cell_voltage_v', 'crossover_rate_mol_s'])
    fit.add_argument('--population', type=int, default=50)
    fit.add_argument('--generations', type=int, default=50, help='Offspring generations after initialization')
    fit.add_argument('--repeats', type=int, default=5)
    fit.add_argument('--subset-fraction', type=float, default=.8)
    fit.add_argument('--correlation-threshold', type=float, default=.9)
    fit.add_argument('--crossover', type=float, default=.5)
    fit.add_argument('--mutation', type=float, default=.1)
    fit.add_argument('--min-depth', type=int, default=3)
    fit.add_argument('--max-depth', type=int, default=12)
    fit.add_argument('--max-nodes', type=int, default=255)
    fit.add_argument('--tournament-size', type=int, default=3)
    fit.add_argument('--top-k', type=int, default=3)
    fit.add_argument('--distance-scaling', choices=['none', 'target-std'], default='none')
    fit.add_argument('--seed', type=int, default=42)
    evaluate = commands.add_parser('evaluate', help='Evaluate a saved proxy on physical design rows')
    evaluate.add_argument('model', type=Path, help='A generated proxy.json or best_so_far.json')
    evaluate.add_argument('--input', type=Path, required=True)
    evaluate.add_argument('--output', type=Path, required=True)
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'fit':
            if __package__:
                from .proxy_generation import run
            else:
                from proxy_generation import run
            results = run(args)
            print(f'Wrote {len(results)} proxies; report: {args.output / "report.md"}')
        else:
            import pandas as pd
            if __package__:
                from .proxy_functions import ProxyFunction
            else:
                from proxy_functions import ProxyFunction
            model = ProxyFunction(args.model)
            data = pd.read_csv(args.input)
            missing = set(model.input_columns)-set(data.columns)
            if missing:
                raise ValueError(f'Missing design columns: {sorted(missing)}')
            result = data[model.input_columns].copy()
            if 'sample_id' in data:
                result.insert(0, 'sample_id', data.sample_id)
            result['proxy_normalized'] = model(result[model.input_columns].to_numpy())
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x') as stream:
                result.to_csv(stream, index=False)
            print(f'Wrote {len(result)} synthetic proxy evaluations to {args.output}')
    except KeyboardInterrupt:
        print('Proxy search interrupted. See history.csv and best_so_far.json for completed work.', file=sys.stderr)
        return 130
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f'Proxy error: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
