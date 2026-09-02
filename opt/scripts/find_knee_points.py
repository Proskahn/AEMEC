#!/usr/bin/env python3
"""Find two Pareto knee points and regenerate the annotated Pareto plot."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Sequence

# Allow this file to remain directly executable from the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aemec_opt.cli import AEMEC_REPORT
from aemec_opt.engine import OptimizationError
from aemec_opt.reporting import TrialRecord, write_pareto_artifacts


def load_completed_records(results_path: Path) -> list[TrialRecord]:
    """Load the columns needed for knee selection from an AEMEC results CSV."""
    if not results_path.is_file():
        raise OptimizationError(f"Optimization results not found: {results_path}")
    with results_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise OptimizationError(f"Optimization results are empty: {results_path}")

    required = {
        "trial",
        AEMEC_REPORT.parameter_name,
        *AEMEC_REPORT.objective_names,
    }
    missing = required.difference(rows[0])
    if missing:
        raise OptimizationError(
            "Optimization results are missing columns: "
            + ", ".join(sorted(missing))
        )

    records: list[TrialRecord] = []
    for row in rows:
        try:
            trial = int(row["trial"])
            parameter = float(row[AEMEC_REPORT.parameter_name])
            objectives = tuple(
                float(row[name]) for name in AEMEC_REPORT.objective_names
            )
        except (TypeError, ValueError) as exc:
            raise OptimizationError(
                "Optimization results contain an invalid trial or numeric value"
            ) from exc
        if not math.isfinite(parameter) or not all(
            math.isfinite(value) for value in objectives
        ):
            raise OptimizationError(
                "Optimization results contain a non-finite parameter or objective"
            )
        records.append(TrialRecord(trial, parameter, objectives, {}))
    return sorted(records, key=lambda record: record.number)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study_dir",
        type=Path,
        help="Study directory containing optimization_results.csv.",
    )
    parser.add_argument(
        "--bend-angle-threshold-deg",
        type=float,
        default=0.0,
        metavar="DEGREES",
        help=(
            "Minimum positive bend angle required for a bend-angle knee "
            "(default: 0)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    study_dir = args.study_dir.resolve()
    try:
        records = load_completed_records(study_dir / "optimization_results.csv")
        knees = write_pareto_artifacts(
            records,
            study_dir,
            AEMEC_REPORT,
            ("minimize", "minimize"),
            args.bend_angle_threshold_deg,
        )
    except (OSError, OptimizationError) as exc:
        parser.exit(1, f"error: {exc}\n")

    by_method = {knee.method: knee for knee in knees}
    chebyshev = by_method["chebyshev"]
    print(
        "Chebyshev knee: "
        f"trial {chebyshev.record.number}, "
        f"{AEMEC_REPORT.parameter_name}={chebyshev.record.parameter_value:g}, "
        f"normalized distance={chebyshev.metric_value:.8g}"
    )
    bend_angle = by_method.get("bend_angle")
    if bend_angle is None:
        print(
            "Bend-angle knee: none; fewer than three distinct trade-off points "
            "are available or the maximum positive angle does not exceed "
            f"{args.bend_angle_threshold_deg:g} degrees"
        )
    else:
        print(
            "Bend-angle knee: "
            f"trial {bend_angle.record.number}, "
            f"{AEMEC_REPORT.parameter_name}={bend_angle.record.parameter_value:g}, "
            f"angle={bend_angle.metric_value:.8g} degrees"
        )
    print(f"Wrote knee points: {study_dir / 'knee_points.csv'}")
    print(f"Updated Pareto plot: {study_dir / 'pareto_front.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
