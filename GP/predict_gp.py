"""Predict both AEMEC objectives using the two saved GP pipelines."""
import argparse
import json
from pathlib import Path

import pandas as pd

if __package__:
    from .models import predict_saved
else:
    from models import predict_saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path, help='Designs with all three input columns')
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-extrapolation', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError('Prediction output already exists; choose a new --output')
        metadata = json.loads((args.model_dir/'metadata.json').read_text())
        predictions = predict_saved(args.model_dir, pd.read_csv(args.csv), metadata, args.allow_extrapolation)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output, index=False)
        print(f'Wrote {len(predictions)} predictions to {args.output}')
    except (ValueError, OSError) as exc:
        parser.exit(2, f'GP error: {exc}\n')


if __name__ == '__main__':
    main()
