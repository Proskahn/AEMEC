#!/usr/bin/env python3
"""Regenerate summary CSV and figures from saved experiment time series."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from crossover_experiments.parsing import (  # noqa: E402
    read_timeseries_csv,
    summarize_samples,
)
from crossover_experiments.config import case_label  # noqa: E402
from crossover_experiments.project import DEFAULT_OUTPUT_DIR, OptimizationError  # noqa: E402
from crossover_experiments.reporting import write_reports  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate crossover plots from archived CSV time series."
    )
    parser.add_argument("result_dir", nargs="?", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result_dir = args.result_dir.resolve()
    manifest_path = result_dir / "experiment.json"
    if not manifest_path.is_file():
        print(f"error: experiment manifest not found: {manifest_path}")
        return 2
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest["settings"]["config"]
        final_window_s = float(config["final_window_s"])
        target_a_cm2 = float(config["target_current_density_a_m2"]) / 1.0e4
        thicknesses = [float(value) for value in config["thicknesses_um"]]
        samples_by_thickness = {}
        summaries = []
        for thickness in thicknesses:
            label = case_label(thickness)
            data_path = result_dir / "data" / f"{label}_timeseries.csv"
            if not data_path.is_file():
                continue
            samples = read_timeseries_csv(data_path)
            samples_by_thickness[thickness] = samples
            summaries.append(summarize_samples(thickness, samples, final_window_s))
        if not summaries:
            raise OptimizationError(f"No completed time-series CSV files found in {result_dir / 'data'}")
        write_reports(result_dir, samples_by_thickness, summaries, target_a_cm2)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OptimizationError) as exc:
        print(f"error: {exc}")
        return 2
    print(f"Wrote {result_dir / 'summary.csv'}")
    print(f"Wrote {result_dir / 'timeseries.png'}")
    print(f"Wrote {result_dir / 'thickness_summary.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
