#!/usr/bin/env python3
"""Backfill per-trial polarization-curve CSV and PNG files from solver logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from aemec_case import (
    AemecEvaluationConfig,
    write_polarization_curve_csv,
    write_polarization_curve_plot,
)


def _interpolation_markers(
    results_path: Path,
) -> dict[int, tuple[float, float, float]]:
    if not results_path.is_file():
        return {}
    markers: dict[int, tuple[float, float, float]] = {}
    with results_path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            try:
                markers[int(row["trial"])] = (
                    float(row["lower_voltage_v"]),
                    float(row["upper_voltage_v"]),
                    float(row["interpolation_fraction"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
    return markers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create trial_XXXX_polarization_curve.csv and .png beside every "
            "existing trial_XXXX_solver.log in an optimization study."
        )
    )
    parser.add_argument(
        "study_dir",
        type=Path,
        help="Study directory containing optimization_results.csv and logs/.",
    )
    parser.add_argument("--stability-samples", type=int, default=5)
    parser.add_argument(
        "--target-current-density-a-m2", type=float, default=10_000.0
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing curve CSV and PNG files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    study_dir = args.study_dir.resolve()
    logs_dir = study_dir / "logs"
    if not logs_dir.is_dir():
        raise SystemExit(f"error: logs directory not found: {logs_dir}")
    config = AemecEvaluationConfig(
        target_current_density_a_m2=args.target_current_density_a_m2,
        stability_samples=args.stability_samples,
    )
    config.validate()
    markers = _interpolation_markers(study_dir / "optimization_results.csv")
    csv_written = plots_written = unchanged = 0
    for solver_log in sorted(logs_dir.glob("trial_*_solver.log")):
        trial_text = solver_log.name.removeprefix("trial_").removesuffix(
            "_solver.log"
        )
        try:
            trial_number = int(trial_text)
        except ValueError:
            continue
        output_path = logs_dir / f"trial_{trial_number:04d}_polarization_curve.csv"
        plot_path = logs_dir / f"trial_{trial_number:04d}_polarization_curve.png"
        write_csv = args.overwrite or not output_path.exists()
        write_plot = args.overwrite or not plot_path.exists()
        if not write_csv and not write_plot:
            unchanged += 1
            continue
        if write_csv:
            point_count = write_polarization_curve_csv(
                output_path,
                solver_log.read_text(encoding="utf-8", errors="replace"),
                config,
                markers.get(trial_number),
            )
            print(
                f"trial {trial_number}: {point_count} voltage points -> "
                f"{output_path}"
            )
            csv_written += 1
        if write_plot:
            point_count = write_polarization_curve_plot(
                output_path,
                plot_path,
                f"Trial {trial_number} Polarization Curve",
            )
            print(f"trial {trial_number}: {point_count} plotted points -> {plot_path}")
            plots_written += 1
    print(
        f"CSV written: {csv_written}; plots written: {plots_written}; "
        f"unchanged: {unchanged}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
