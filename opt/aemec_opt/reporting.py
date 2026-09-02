"""Reusable CSV and two-objective Pareto reporting.

The report layout is configured by the application.  This module has no
OpenFOAM or AEMEC dependency.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna

from .engine import OptimizationError, pareto_mask
from .knee import find_bend_angle_knee, find_chebyshev_knee


@dataclass(frozen=True)
class TrialRecord:
    number: int
    parameter_value: float
    objective_values: tuple[float, ...]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class ReportSpec:
    parameter_name: str
    parameter_label: str
    objective_names: tuple[str, str]
    objective_labels: tuple[str, str]
    title: str
    metadata_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class KneePointRecord:
    """A selected trial and the normalized metric that selected it."""

    method: str
    record: TrialRecord
    normalized_objectives: tuple[float, float]
    metric_name: str
    metric_value: float


def completed_records(
    study: optuna.study.Study, parameter_name: str
) -> list[TrialRecord]:
    records: list[TrialRecord] = []
    for trial in study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
    ):
        if parameter_name not in trial.params or not trial.values:
            continue
        records.append(
            TrialRecord(
                number=trial.number,
                parameter_value=float(trial.params[parameter_name]),
                objective_values=tuple(float(value) for value in trial.values),
                metadata=dict(trial.user_attrs),
            )
        )
    return sorted(records, key=lambda record: record.number)


def record_pareto_mask(records: Sequence[TrialRecord], directions: Sequence[str]) -> list[bool]:
    return pareto_mask([record.objective_values for record in records], directions)


def find_pareto_knee_points(
    records: Sequence[TrialRecord],
    directions: Sequence[str],
    bend_angle_threshold_degrees: float = 0.0,
) -> tuple[KneePointRecord, ...]:
    """Find normalized Chebyshev and bend-angle knees on the Pareto front."""
    if len(directions) != 2 or any(
        len(record.objective_values) != 2 for record in records
    ):
        raise OptimizationError("Knee-point reporting requires exactly two objectives")
    mask = record_pareto_mask(records, directions)
    pareto_records = [
        record for record, is_pareto in zip(records, mask) if is_pareto
    ]
    objectives = [record.objective_values for record in pareto_records]
    selected: list[KneePointRecord] = []

    chebyshev = find_chebyshev_knee(objectives, directions)
    if chebyshev is not None:
        normalized = chebyshev.normalized_objectives
        selected.append(
            KneePointRecord(
                "chebyshev",
                pareto_records[chebyshev.index],
                (normalized[0], normalized[1]),
                "normalized_chebyshev_distance",
                chebyshev.distance,
            )
        )

    bend_angle = find_bend_angle_knee(
        objectives,
        directions,
        bend_angle_threshold_degrees,
    )
    if bend_angle is not None:
        selected.append(
            KneePointRecord(
                "bend_angle",
                pareto_records[bend_angle.index],
                bend_angle.normalized_objectives,
                "bend_angle_degrees",
                bend_angle.angle_degrees,
            )
        )
    return tuple(selected)


def _csv_value(value: object) -> object:
    return "" if value is None else value


def write_pareto_artifacts(
    records: Sequence[TrialRecord],
    output_dir: Path,
    spec: ReportSpec,
    directions: Sequence[str],
    bend_angle_threshold_degrees: float = 0.0,
) -> tuple[KneePointRecord, ...]:
    """Write the Pareto CSV, knee-point CSV, and annotated Pareto plot."""
    if len(spec.objective_names) != 2 or len(spec.objective_labels) != 2:
        raise OptimizationError("The built-in report currently requires two objectives")
    if len(directions) != 2:
        raise OptimizationError("The built-in report currently requires two objectives")
    output_dir.mkdir(parents=True, exist_ok=True)
    mask = record_pareto_mask(records, directions)
    pareto_records = sorted(
        (record for record, is_pareto in zip(records, mask) if is_pareto),
        key=lambda record: record.objective_values[0],
    )
    knees = find_pareto_knee_points(
        pareto_records,
        directions,
        bend_angle_threshold_degrees,
    )

    pareto_path = output_dir / "pareto_front.csv"
    with pareto_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["trial", spec.parameter_name, *spec.objective_names])
        for record in pareto_records:
            writer.writerow(
                [record.number, record.parameter_value, *record.objective_values]
            )

    knee_path = output_dir / "knee_points.csv"
    with knee_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "method",
                "trial",
                spec.parameter_name,
                *spec.objective_names,
                "normalized_objective_1",
                "normalized_objective_2",
                "metric",
                "metric_value",
            ]
        )
        for knee in knees:
            writer.writerow(
                [
                    knee.method,
                    knee.record.number,
                    knee.record.parameter_value,
                    *knee.record.objective_values,
                    *knee.normalized_objectives,
                    knee.metric_name,
                    knee.metric_value,
                ]
            )

    if not records:
        return knees
    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    scatter = ax.scatter(
        [record.objective_values[0] for record in records],
        [record.objective_values[1] for record in records],
        c=[record.parameter_value for record in records],
        cmap="viridis",
        s=48,
        alpha=0.78,
        edgecolors="none",
        label="Evaluated designs",
    )
    ax.plot(
        [record.objective_values[0] for record in pareto_records],
        [record.objective_values[1] for record in pareto_records],
        color="tab:red",
        marker="o",
        markersize=5,
        linewidth=1.7,
        label="Pareto front",
    )
    knee_styles = {
        "chebyshev": {
            "marker": "*",
            "s": 180,
            "facecolors": "tab:blue",
            "edgecolors": "white",
            "label": "Chebyshev knee",
        },
        "bend_angle": {
            "marker": "D",
            "s": 105,
            "facecolors": "none",
            "edgecolors": "darkorange",
            "label": "Bend-angle knee",
        },
    }
    for knee in knees:
        style = knee_styles[knee.method]
        ax.scatter(
            [knee.record.objective_values[0]],
            [knee.record.objective_values[1]],
            linewidths=1.8,
            zorder=5,
            **style,
        )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(spec.parameter_label)
    ax.set_xlabel(spec.objective_labels[0])
    ax.set_ylabel(spec.objective_labels[1])
    ax.set_title(spec.title)
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.savefig(output_dir / "pareto_front.png", dpi=220)
    plt.close(fig)
    return knees


def write_results(
    records: Sequence[TrialRecord],
    output_dir: Path,
    spec: ReportSpec,
    directions: Sequence[str],
    bend_angle_threshold_degrees: float = 0.0,
) -> None:
    """Write all evaluations and the Pareto and knee-point artifacts."""
    if len(spec.objective_names) != 2 or len(spec.objective_labels) != 2:
        raise OptimizationError("The built-in report currently requires two objectives")
    output_dir.mkdir(parents=True, exist_ok=True)
    mask = record_pareto_mask(records, directions)

    results_path = output_dir / "optimization_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "trial",
                spec.parameter_name,
                *spec.objective_names,
                *spec.metadata_columns,
                "is_pareto",
            ]
        )
        for record, is_pareto in zip(records, mask):
            writer.writerow(
                [
                    record.number,
                    record.parameter_value,
                    *record.objective_values,
                    *[_csv_value(record.metadata.get(key)) for key in spec.metadata_columns],
                    is_pareto,
                ]
            )

    write_pareto_artifacts(
        records,
        output_dir,
        spec,
        directions,
        bend_angle_threshold_degrees,
    )
