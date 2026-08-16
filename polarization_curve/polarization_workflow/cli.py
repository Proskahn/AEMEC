"""Command-line interface for the former AEMEC polarization curve."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Sequence

from .project import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_CASE,
    DEFAULT_WORK_DIR,
    OptimizationError,
)
from .runner import PolarizationCurveRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the retained AEMEC 1.3--2.3 V polarization table for 165 s "
            "and generate the polarization, voltage-loss, and crossover plots."
        )
    )
    parser.add_argument("--source-case", type=Path, default=DEFAULT_SOURCE_CASE)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mesh-command", default="make mesh")
    parser.add_argument("--solver-command", default="openFuelCell")
    parser.add_argument(
        "--active-area-cm2",
        type=float,
        default=0.8,
        help="Fallback active area for legacy logs (default: 0.8 cm2)",
    )
    parser.add_argument("--timeout-minutes", type=float, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing result directory with a fresh run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and validate the isolated 165 s work case without OpenFOAM",
    )
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Regenerate CSV/PNG outputs from output-dir/logs/solver.log",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        runner = PolarizationCurveRunner(
            source_case=args.source_case,
            work_dir=args.work_dir,
            output_dir=args.output_dir,
            mesh_command=shlex.split(args.mesh_command),
            solver_command=shlex.split(args.solver_command),
            timeout_s=(
                None
                if args.timeout_minutes is None
                else 60.0 * args.timeout_minutes
            ),
            active_area_cm2=args.active_area_cm2,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            postprocess_only=args.postprocess_only,
        )
        return runner.run()
    except (OptimizationError, OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
