"""Command-line composition of the generic engine and the AEMEC adapter."""

from __future__ import annotations

import argparse
import math
import shlex
import sys
from pathlib import Path
from typing import Sequence

from aemec_case import (
    AemecEvaluationConfig,
    AemecOpenFoamEvaluator,
    DEFAULT_CROSSOVER_STABILITY_RELATIVE_TOLERANCE,
    DEFAULT_CURRENT_RELATIVE_TOLERANCE,
    DEFAULT_STABILITY_SAMPLES,
    DEFAULT_TARGET_CURRENT_DENSITY_A_M2,
    DEFAULT_VOLTAGE_STABILITY_TOLERANCE_V,
    case_fingerprint,
    validate_path_layout,
)
from optimization_lib import (
    OptimizationError,
    SearchConfig,
    completed_trial_count,
    create_or_load_study,
    run_study,
    study_artifact_directory,
)
from reporting import ReportSpec, completed_records, write_results


DEFAULT_ITERATIONS = 50
DEFAULT_MIN_THICKNESS_UM = 10.0
DEFAULT_MAX_THICKNESS_UM = 100.0
DEFAULT_STARTUP_TRIALS = 10
DEFAULT_SEED = 42
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3
LEGACY_DEFAULT_RUN_MODE = "fast"
LEGACY_DEFAULT_SOLVER_ITERATIONS = 250
LEGACY_DEFAULT_ITERATION_CLOCK_STEP = 1.0
LEGACY_DEFAULT_TARGET_HOLD_DURATION_S = 30.0
LEGACY_DEFAULT_CONTROLLER_STEP_TOLERANCE_V = 0.001
PARAMETER_NAME = "membrane_thickness_um"

AEMEC_REPORT = ReportSpec(
    parameter_name=PARAMETER_NAME,
    parameter_label="Membrane thickness [um]",
    objective_names=("cell_voltage_v", "crossover_rate_mol_s"),
    objective_labels=("Cell voltage at 1 A/cm2 [V]", "H2 crossover rate [mol/s]"),
    title="AEMEC membrane-thickness Pareto front",
    metadata_columns=(
        "target_current_density_a_m2",
        "interpolation_fraction",
        "lower_time_s",
        "lower_current_density_a_m2",
        "lower_voltage_v",
        "upper_time_s",
        "upper_current_density_a_m2",
        "upper_voltage_v",
        "voltage_hold_count",
        "sweep_end_s",
        "solver_clock_step",
        "solver_outer_iterations",
        "duration_s",
        "solver_log",
        "polarization_curve_csv",
        "polarization_curve_plot",
    ),
)


def parse_command(value: str) -> tuple[str, ...]:
    command = tuple(shlex.split(value))
    if not command:
        raise argparse.ArgumentTypeError("command cannot be empty")
    return command


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run a resumable AEMEC membrane-thickness Pareto optimization."
    )
    parser.add_argument("--case", type=Path, default=root / "run/AEMEC")
    parser.add_argument("--output-dir", type=Path, default=root / "opt/results")
    parser.add_argument("--work-dir", type=Path, default=root / "opt/work")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Completed CFD evaluations.")
    parser.add_argument("--min-thickness-um", type=float, default=DEFAULT_MIN_THICKNESS_UM)
    parser.add_argument("--max-thickness-um", type=float, default=DEFAULT_MAX_THICKNESS_UM)
    parser.add_argument("--startup-trials", type=int, default=DEFAULT_STARTUP_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-consecutive-failures", type=int, default=DEFAULT_MAX_CONSECUTIVE_FAILURES)
    parser.add_argument("--study-name", default="aemec-membrane-thickness")
    parser.add_argument("--target-current-density-a-m2", type=float, default=DEFAULT_TARGET_CURRENT_DENSITY_A_M2)
    parser.add_argument("--current-relative-tolerance", type=float, default=DEFAULT_CURRENT_RELATIVE_TOLERANCE, help="Maximum current coefficient of variation in each objective voltage-hold window.")
    parser.add_argument("--run-mode", choices=("fast", "ramp"), default=LEGACY_DEFAULT_RUN_MODE, help="Legacy compatibility option; the optimizer always runs the complete voltage sweep.")
    parser.add_argument("--solver-iterations", type=int, default=LEGACY_DEFAULT_SOLVER_ITERATIONS, help="Legacy compatibility option; ignored by potentiostatic sweep evaluations.")
    parser.add_argument("--iteration-clock-step", type=float, default=LEGACY_DEFAULT_ITERATION_CLOCK_STEP, help="Legacy compatibility option; the sweep uses controlDict.run deltaT.")
    parser.add_argument("--target-hold-duration-s", type=float, default=LEGACY_DEFAULT_TARGET_HOLD_DURATION_S, help="Legacy compatibility option; voltage hold times come from the voltage table.")
    parser.add_argument("--stability-samples", type=int, default=DEFAULT_STABILITY_SAMPLES)
    parser.add_argument("--voltage-stability-tolerance-v", type=float, default=DEFAULT_VOLTAGE_STABILITY_TOLERANCE_V)
    parser.add_argument("--crossover-stability-relative-tolerance", type=float, default=DEFAULT_CROSSOVER_STABILITY_RELATIVE_TOLERANCE)
    parser.add_argument("--controller-step-tolerance-v", type=float, default=LEGACY_DEFAULT_CONTROLLER_STEP_TOLERANCE_V, help="Legacy compatibility option; no galvanostatic controller is used.")
    parser.add_argument("--mesh-command", type=parse_command, default=("make", "mesh"))
    parser.add_argument("--solver-command", type=parse_command, default=("openFuelCell",))
    parser.add_argument("--timeout-minutes", type=float, default=None)
    return parser


def _evaluation_config(args: argparse.Namespace) -> AemecEvaluationConfig:
    return AemecEvaluationConfig(
        target_current_density_a_m2=args.target_current_density_a_m2,
        current_relative_tolerance=args.current_relative_tolerance,
        stability_samples=args.stability_samples,
        voltage_stability_tolerance_v=args.voltage_stability_tolerance_v,
        crossover_stability_relative_tolerance=args.crossover_stability_relative_tolerance,
    )


def _search_config(args: argparse.Namespace) -> SearchConfig:
    return SearchConfig(
        parameter_name=PARAMETER_NAME,
        lower_bound=args.min_thickness_um,
        upper_bound=args.max_thickness_um,
        completed_evaluations=args.iterations,
        startup_trials=args.startup_trials,
        seed=args.seed,
        max_consecutive_failures=args.max_consecutive_failures,
    )


def validate_args(args: argparse.Namespace) -> tuple[SearchConfig, AemecEvaluationConfig]:
    search = _search_config(args)
    search.validate()
    evaluation = _evaluation_config(args)
    evaluation.validate()
    if args.timeout_minutes is not None and (
        not math.isfinite(args.timeout_minutes) or args.timeout_minutes <= 0
    ):
        raise OptimizationError("--timeout-minutes must be finite and positive")
    return search, evaluation


def run_optimization(args: argparse.Namespace):
    search, evaluation_config = validate_args(args)
    output_dir = study_artifact_directory(args.output_dir, args.study_name)
    work_case = args.work_dir.resolve() / "case"
    validate_path_layout(args.case, work_case, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    application_settings = {
        "source_case": str(args.case.resolve()),
        "case_fingerprint": case_fingerprint(args.case.resolve()),
        "evaluation": evaluation_config.resume_settings(),
        "mesh_command": list(args.mesh_command),
        "solver_command": list(args.solver_command),
    }
    study = create_or_load_study(
        storage_path=output_dir / "optimization.db",
        study_name=args.study_name,
        config=search,
        application_settings=application_settings,
    )
    evaluator = AemecOpenFoamEvaluator(
        source_case=args.case,
        work_case=work_case,
        logs_dir=output_dir / "logs",
        mesh_command=args.mesh_command,
        solver_command=args.solver_command,
        config=evaluation_config,
        timeout_s=None if args.timeout_minutes is None else args.timeout_minutes * 60.0,
    )

    def evaluate(trial_number: int, thickness_um: float):
        result = evaluator.evaluate(trial_number, thickness_um)
        objective = result.objective
        print(
            f"  interpolated voltage={objective.cell_voltage_v:.8g} V, "
            f"crossover={objective.crossover_rate_mol_s:.8g} mol/s at "
            f"{objective.target_current_density_a_m2:.8g} A/m2\n"
            f"  polarization data={result.polarization_curve_csv}\n"
            f"  polarization plot={result.polarization_curve_plot}",
            flush=True,
        )
        return result.as_objective_result()

    def checkpoint(checkpoint_study, _finished_trial) -> None:
        write_results(
            completed_records(checkpoint_study, PARAMETER_NAME),
            output_dir,
            AEMEC_REPORT,
            search.directions,
        )

    print(
        f"Running until {search.completed_evaluations} completed CFD evaluations; "
        f"{completed_trial_count(study)} already complete. Results: {output_dir}",
        flush=True,
    )
    run_study(study=study, config=search, evaluate=evaluate, checkpoint=checkpoint)
    records = completed_records(study, PARAMETER_NAME)
    write_results(records, output_dir, AEMEC_REPORT, search.directions)
    print(f"\nCompleted CFD evaluations: {len(records)}")
    print(f"Results CSV: {output_dir / 'optimization_results.csv'}")
    print(f"Pareto plot: {output_dir / 'pareto_front.png'}")
    return study


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_optimization(args)
    except (OptimizationError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
