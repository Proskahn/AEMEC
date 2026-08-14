"""Model-neutral multi-objective optimization primitives.

This module deliberately knows nothing about OpenFOAM, membranes, plotting, or
the meaning of an objective.  An application supplies an evaluator that maps a
continuous design value to objective values and JSON-compatible metadata.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import optuna


class OptimizationError(RuntimeError):
    """Raised when an evaluation cannot yield trustworthy objectives."""


@dataclass(frozen=True)
class SearchConfig:
    parameter_name: str
    lower_bound: float
    upper_bound: float
    completed_evaluations: int
    startup_trials: int
    seed: int
    max_consecutive_failures: int
    directions: tuple[str, ...] = ("minimize", "minimize")

    def validate(self) -> None:
        if not self.parameter_name:
            raise OptimizationError("The design-parameter name cannot be empty")
        if (
            not math.isfinite(self.lower_bound)
            or not math.isfinite(self.upper_bound)
            or self.lower_bound >= self.upper_bound
        ):
            raise OptimizationError("Search bounds must be finite and ordered")
        if self.completed_evaluations <= 0:
            raise OptimizationError("Completed-evaluation budget must be positive")
        if self.startup_trials < 2:
            raise OptimizationError("At least two startup trials are required")
        if self.max_consecutive_failures <= 0:
            raise OptimizationError("Failure limit must be positive")
        if not self.directions or any(
            direction not in {"minimize", "maximize"}
            for direction in self.directions
        ):
            raise OptimizationError("Objectives must be minimized or maximized")


@dataclass(frozen=True)
class ObjectiveResult:
    values: tuple[float, ...]
    metadata: Mapping[str, object]

    def validate(self, objective_count: int) -> None:
        if len(self.values) != objective_count:
            raise OptimizationError(
                f"Evaluator returned {len(self.values)} objectives; expected "
                f"{objective_count}"
            )
        if not all(math.isfinite(value) for value in self.values):
            raise OptimizationError("Evaluator returned a non-finite objective")


Evaluator = Callable[[int, float], ObjectiveResult]
Checkpoint = Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]


def study_artifact_directory(output_root: Path, study_name: str) -> Path:
    """Return a filesystem-safe, case-insensitive unique study directory."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", study_name).strip("-.") or "study"
    suffix = hashlib.sha256(study_name.encode("utf-8")).hexdigest()[:8]
    return output_root.resolve() / f"{slug}-{suffix}"


def study_settings(config: SearchConfig, application_settings: Mapping[str, object]) -> dict[str, object]:
    """Combine engine identity with settings supplied by the application adapter."""
    return {
        "optimization_engine_schema_version": 1,
        "parameter_name": config.parameter_name,
        "lower_bound": config.lower_bound,
        "upper_bound": config.upper_bound,
        "directions": list(config.directions),
        "startup_trials": config.startup_trials,
        "seed": config.seed,
        "application": dict(application_settings),
    }


def validate_or_store_settings(study: optuna.study.Study, settings: Mapping[str, object]) -> None:
    stored = study.user_attrs.get("optimization_settings")
    if stored is None:
        study.set_user_attr("optimization_settings", dict(settings))
    elif stored != settings:
        raise OptimizationError(
            "The existing study was created with different engine or application "
            "settings. Use a new study name or output directory."
        )


def create_or_load_study(
    *,
    storage_path: Path,
    study_name: str,
    config: SearchConfig,
    application_settings: Mapping[str, object],
) -> optuna.study.Study:
    """Create a resumable Optuna study and seed its initial space-filling points."""
    config.validate()
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path}"
    resume_trial_count = 0
    if storage_path.is_file():
        try:
            existing = optuna.load_study(study_name=study_name, storage=storage_url)
        except KeyError:
            pass
        else:
            resume_trial_count = len(existing.trials)

    sampler = optuna.samplers.TPESampler(
        # RNG state is not stored by Optuna. The offset prevents replaying a
        # previous proposal sequence after a process restart.
        seed=config.seed + resume_trial_count,
        n_startup_trials=min(config.startup_trials, config.completed_evaluations),
    )
    study = optuna.create_study(
        study_name=study_name,
        directions=config.directions,
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True,
    )
    validate_or_store_settings(study, study_settings(config, application_settings))

    if not study.trials:
        count = min(config.startup_trials, config.completed_evaluations)
        denominator = max(1, count - 1)
        for index in range(count):
            value = config.lower_bound + (
                config.upper_bound - config.lower_bound
            ) * index / denominator
            study.enqueue_trial({config.parameter_name: value})
    return study


def completed_trial_count(study: optuna.study.Study) -> int:
    return sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )


def run_study(
    *,
    study: optuna.study.Study,
    config: SearchConfig,
    evaluate: Evaluator,
    checkpoint: Checkpoint | None = None,
    progress: Callable[[str], None] = print,
) -> optuna.study.Study:
    """Evaluate until the requested number of *successful* trials exists."""
    config.validate()
    completed = completed_trial_count(study)
    if completed > config.completed_evaluations:
        raise OptimizationError(
            f"Study already has {completed} completed evaluations, exceeding the "
            f"requested {config.completed_evaluations}"
        )

    last_finished: optuna.trial.FrozenTrial | None = None

    def objective(trial: optuna.trial.Trial) -> tuple[float, ...]:
        value = trial.suggest_float(
            config.parameter_name, config.lower_bound, config.upper_bound
        )
        progress(f"\nTrial {trial.number}: {config.parameter_name} = {value:.8g}")
        try:
            result = evaluate(trial.number, value)
        except (OptimizationError, OSError) as exc:
            trial.set_user_attr("failure", str(exc))
            progress(f"  trial failed: {exc}")
            if isinstance(exc, OptimizationError):
                raise
            raise OptimizationError(str(exc)) from exc
        result.validate(len(config.directions))
        for key, metadata_value in result.metadata.items():
            trial.set_user_attr(key, metadata_value)
        progress("  objectives=" + ", ".join(f"{item:.8g}" for item in result.values))
        return result.values

    def after_trial(
        checkpoint_study: optuna.study.Study,
        finished_trial: optuna.trial.FrozenTrial,
    ) -> None:
        nonlocal last_finished
        last_finished = finished_trial
        if checkpoint:
            checkpoint(checkpoint_study, finished_trial)

    consecutive_failures = 0
    while completed < config.completed_evaluations:
        last_finished = None
        study.optimize(
            objective,
            n_trials=1,
            n_jobs=1,
            callbacks=[after_trial],
            catch=(OptimizationError,),
            show_progress_bar=False,
        )
        if last_finished is None:
            raise OptimizationError("Optuna returned without finishing a trial")
        if last_finished.state == optuna.trial.TrialState.COMPLETE:
            completed += 1
            consecutive_failures = 0
            progress(
                f"Progress: {completed}/{config.completed_evaluations} completed evaluations"
            )
        else:
            consecutive_failures += 1
            if consecutive_failures >= config.max_consecutive_failures:
                failure = last_finished.user_attrs.get("failure", "unknown trial error")
                raise OptimizationError(
                    f"Aborting after {consecutive_failures} consecutive failed trials. "
                    f"Last error: {failure}"
                )
    return study


def pareto_mask(
    objective_vectors: Sequence[Sequence[float]], directions: Sequence[str]
) -> list[bool]:
    """Return a non-dominated mask for arbitrary minimize/maximize objectives."""
    if any(len(values) != len(directions) for values in objective_vectors):
        raise OptimizationError("Objective vectors and directions have different sizes")
    result: list[bool] = []
    for index, candidate in enumerate(objective_vectors):
        dominated = False
        for other_index, other in enumerate(objective_vectors):
            if index == other_index:
                continue
            no_worse = all(
                (other[dimension] <= candidate[dimension])
                if directions[dimension] == "minimize"
                else (other[dimension] >= candidate[dimension])
                for dimension in range(len(directions))
            )
            strictly_better = any(
                (other[dimension] < candidate[dimension])
                if directions[dimension] == "minimize"
                else (other[dimension] > candidate[dimension])
                for dimension in range(len(directions))
            )
            if no_worse and strictly_better:
                dominated = True
                break
        result.append(not dominated)
    return result
