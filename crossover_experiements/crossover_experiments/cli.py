"""Command-line interface for crossover experiments."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_THICKNESSES_UM, ExperimentConfig
from .project import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_CASE,
    DEFAULT_WORK_DIR,
    OptimizationError,
)
from .runner import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 20 s AEMEC membrane-thickness crossover study at a fixed "
            "current density of 1 A/cm²."
        )
    )
    parser.add_argument(
        "--thicknesses",
        nargs="+",
        type=float,
        default=list(DEFAULT_THICKNESSES_UM),
        metavar="UM",
        help="Membrane thicknesses in µm (default: 20 40 60 80)",
    )
    parser.add_argument("--source-case", type=Path, default=DEFAULT_SOURCE_CASE)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mesh-command",
        default="make mesh",
        help="Command run in each clean case before solving (default: make mesh)",
    )
    parser.add_argument(
        "--solver-command",
        default="openFuelCell",
        help="Solver command run in each meshed case (default: openFuelCell)",
    )
    parser.add_argument(
        "--timeout-minutes",
        type=float,
        default=None,
        help="Per-command timeout; default is unlimited",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun completed thicknesses and replace their generated artifacts",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed thickness instead of continuing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Copy and configure all cases, but do not mesh or solve",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout_minutes is not None and args.timeout_minutes <= 0:
        raise SystemExit("error: --timeout-minutes must be positive")
    config = ExperimentConfig(thicknesses_um=tuple(args.thicknesses))
    try:
        runner = ExperimentRunner(
            source_case=args.source_case,
            work_dir=args.work_dir,
            output_dir=args.output_dir,
            mesh_command=shlex.split(args.mesh_command),
            solver_command=shlex.split(args.solver_command),
            config=config,
            timeout_s=(None if args.timeout_minutes is None else args.timeout_minutes * 60.0),
            overwrite=args.overwrite,
            fail_fast=args.fail_fast,
            dry_run=args.dry_run,
        )
        return runner.run()
    except (OptimizationError, ValueError) as exc:
        print(f"error: {exc}")
        return 2

