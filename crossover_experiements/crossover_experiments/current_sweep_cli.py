"""CLI for the hydrogen-crossover current-density sweep."""

from __future__ import annotations

import argparse
import math
import shlex
from pathlib import Path
from typing import Sequence

from .config import CurrentSweepConfig
from .current_sweep_runner import CurrentSweepRunner
from .current_sweep import read_current_sweep_timeseries, summarize_current_sweep
from .current_sweep_reporting import plot_thickness_current_sweeps
from .project import (
    DEFAULT_CURRENT_SWEEP_OUTPUT_DIR,
    DEFAULT_CURRENT_SWEEP_WORK_DIR,
    DEFAULT_SOURCE_CASE,
    OptimizationError,
)


def _targets(step: float) -> tuple[float, ...]:
    if not math.isfinite(step) or step <= 0 or step > 2.0:
        raise OptimizationError("Current-density step must be in (0, 2] A/cm2")
    count = round(2.0 / step)
    if not math.isclose(count * step, 2.0, abs_tol=1.0e-10):
        raise OptimizationError("Current-density step must divide 2 A/cm2 exactly")
    return tuple(round(index * step, 12) for index in range(count + 1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep AEMEC current density from 0 to 2 A/cm² and calculate "
            "area-averaged hydrogen crossover flux density."
        )
    )
    parser.add_argument(
        "--current-step-a-cm2",
        type=float,
        default=0.2,
        help="Current increment in A/cm²; must divide 2 exactly (default: 0.2)",
    )
    thickness = parser.add_mutually_exclusive_group()
    thickness.add_argument("--membrane-thickness-um", type=float, default=30.0)
    thickness.add_argument(
        "--thicknesses", type=float, nargs="+", metavar="UM",
        help="Run and overlay separate current sweeps, e.g. 20 40 60 80",
    )
    parser.add_argument(
        "--membrane-area-m2",
        type=float,
        default=8.0e-5,
        help="Area used to normalize total crossover rate (default: 8e-5 m²)",
    )
    parser.add_argument("--minimum-hold-s", type=float, default=20.0)
    parser.add_argument("--maximum-duration-s", type=float, default=400.0)
    parser.add_argument("--initial-voltage-v", type=float, default=1.3)
    parser.add_argument("--source-case", type=Path, default=DEFAULT_SOURCE_CASE)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_CURRENT_SWEEP_WORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CURRENT_SWEEP_OUTPUT_DIR)
    parser.add_argument("--mesh-command", default="make mesh")
    parser.add_argument("--solver-command", default="openFuelCell")
    parser.add_argument("--timeout-minutes", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout_minutes is not None and args.timeout_minutes <= 0:
        print("error: --timeout-minutes must be positive")
        return 2
    try:
        thicknesses = args.thicknesses or [args.membrane_thickness_um]
        if len(set(thicknesses)) != len(thicknesses):
            raise OptimizationError("Membrane thicknesses must be unique")
        configs = [CurrentSweepConfig(
            current_targets_a_cm2=_targets(args.current_step_a_cm2),
            membrane_thickness_um=thickness,
            membrane_area_m2=args.membrane_area_m2,
            minimum_hold_s=args.minimum_hold_s,
            maximum_duration_s=args.maximum_duration_s,
            initial_voltage_v=args.initial_voltage_v,
        ) for thickness in thicknesses]
        # Validate every case before starting an expensive solve.
        for config in configs:
            config.validate()
        curves = []
        for config in configs:
            suffix = f"{config.membrane_thickness_um:g}um" if args.thicknesses else ""
            output_dir = args.output_dir / suffix
            runner = CurrentSweepRunner(
                source_case=args.source_case,
                work_dir=args.work_dir / suffix,
                output_dir=output_dir,
                mesh_command=shlex.split(args.mesh_command),
                solver_command=shlex.split(args.solver_command),
                config=config,
                timeout_s=(None if args.timeout_minutes is None else 60.0 * args.timeout_minutes),
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            result = runner.run()
            if result:
                return result
            if args.thicknesses and not args.dry_run:
                samples = read_current_sweep_timeseries(output_dir / "data/timeseries.csv")
                curves.append((config.membrane_thickness_um, summarize_current_sweep(samples, config)))
        if curves:
            path = args.output_dir / "membrane_thickness_current_sweep.png"
            plot_thickness_current_sweeps(path, curves)
            print(f"[report] {path}")
        return 0
    except (OptimizationError, ValueError) as exc:
        print(f"error: {exc}")
        return 2

