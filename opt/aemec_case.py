"""AEMEC/OpenFOAM adapter for the generic optimization engine.

This module contains the case-specific science and runtime integration: the
membrane geometry convention, potentiostatic sweep, interpolation contract,
and safe OpenFOAM scratch-case execution.  It does not know how Optuna selects
designs or how Pareto reports are rendered.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from optimization_lib import ObjectiveResult, OptimizationError


# Version 6 derives the 1 A/cm2 objective from a potentiostatic polarization
# sweep. It must not resume studies whose objective came from galvanostatic
# feedback.
OBJECTIVE_SCHEMA_VERSION = 6
DEFAULT_TARGET_CURRENT_DENSITY_A_M2 = 10_000.0
DEFAULT_CURRENT_RELATIVE_TOLERANCE = 0.05
DEFAULT_STABILITY_SAMPLES = 5
DEFAULT_VOLTAGE_STABILITY_TOLERANCE_V = 0.005
DEFAULT_CROSSOVER_STABILITY_RELATIVE_TOLERANCE = 0.02

FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
VERTEX_RE = re.compile(
    rf"\(\s*({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s*\)"
)
TABLE_PAIR_RE = re.compile(rf"\(\s*({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s*\)")
TIME_RE = re.compile(rf"^\s*Time\s*=\s*({FLOAT_PATTERN})\s*$")
BOUNDARY_POINT_RE = re.compile(
    rf"Controlled\s+boundary\s+current.*?current\s+density\s*=\s*"
    rf"({FLOAT_PATTERN})\s*A/m2.*?voltage\s*=\s*({FLOAT_PATTERN})\b",
    re.IGNORECASE,
)
CROSSOVER_RE = re.compile(
    rf"Hydrogen\s+crossover\s+objective:.*?=\s*({FLOAT_PATTERN})\s*mol/s\b",
    re.IGNORECASE,
)
NORMAL_END_RE = re.compile(r"^\s*End\s*$", re.MULTILINE)
FATAL_OUTPUT_RE = re.compile(
    r"FOAM\s+FATAL|Segmentation\s+fault|command\s+not\s+found|Traceback\s+\(most\s+recent",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AemecEvaluationConfig:
    target_current_density_a_m2: float = DEFAULT_TARGET_CURRENT_DENSITY_A_M2
    current_relative_tolerance: float = DEFAULT_CURRENT_RELATIVE_TOLERANCE
    stability_samples: int = DEFAULT_STABILITY_SAMPLES
    voltage_stability_tolerance_v: float = DEFAULT_VOLTAGE_STABILITY_TOLERANCE_V
    crossover_stability_relative_tolerance: float = (
        DEFAULT_CROSSOVER_STABILITY_RELATIVE_TOLERANCE
    )

    def validate(self) -> None:
        if (
            not math.isfinite(self.target_current_density_a_m2)
            or self.target_current_density_a_m2 <= 0
        ):
            raise OptimizationError("Target current-density magnitude must be positive")
        if not 0 < self.current_relative_tolerance < 1:
            raise OptimizationError("Current relative tolerance must be in (0, 1)")
        if self.stability_samples <= 0:
            raise OptimizationError("Stability sample count must be positive")
        if (
            not math.isfinite(self.voltage_stability_tolerance_v)
            or self.voltage_stability_tolerance_v < 0
        ):
            raise OptimizationError("Voltage stability tolerance cannot be negative")
        if not 0 <= self.crossover_stability_relative_tolerance < 1:
            raise OptimizationError("Crossover stability tolerance must be in [0, 1)")

    def resume_settings(self) -> dict[str, object]:
        self.validate()
        return {
            "objective_schema_version": OBJECTIVE_SCHEMA_VERSION,
            "target_current_density_a_m2": self.target_current_density_a_m2,
            "current_relative_tolerance": self.current_relative_tolerance,
            "stability_samples": self.stability_samples,
            "voltage_stability_tolerance_v": self.voltage_stability_tolerance_v,
            "crossover_stability_relative_tolerance": (
                self.crossover_stability_relative_tolerance
            ),
        }


@dataclass(frozen=True)
class GeometryUpdate:
    old_thickness_um: float
    new_thickness_um: float
    center_mm: float
    half_stack_shift_mm: float
    vertex_count: int


@dataclass(frozen=True)
class VoltageHold:
    start_s: float
    end_s: float
    voltage_v: float


@dataclass(frozen=True)
class VoltageSweep:
    holds: tuple[VoltageHold, ...]
    delta_t_s: float
    outer_iterations: int


@dataclass(frozen=True)
class SweepSample:
    time_s: float
    current_density_a_m2: float
    cell_voltage_v: float
    crossover_rate_mol_s: float


@dataclass(frozen=True)
class InterpolatedObjective:
    target_current_density_a_m2: float
    cell_voltage_v: float
    crossover_rate_mol_s: float
    interpolation_fraction: float
    lower_sample: SweepSample
    upper_sample: SweepSample


@dataclass(frozen=True)
class AemecEvaluation:
    objective: InterpolatedObjective
    voltage_sweep: VoltageSweep
    solver_log: Path
    mesh_log: Path
    duration_s: float

    def as_objective_result(self) -> ObjectiveResult:
        objective = self.objective
        lower = objective.lower_sample
        upper = objective.upper_sample
        return ObjectiveResult(
            values=(objective.cell_voltage_v, objective.crossover_rate_mol_s),
            metadata={
                "target_current_density_a_m2": objective.target_current_density_a_m2,
                "interpolation_fraction": objective.interpolation_fraction,
                "lower_time_s": lower.time_s,
                "lower_current_density_a_m2": lower.current_density_a_m2,
                "lower_voltage_v": lower.cell_voltage_v,
                "upper_time_s": upper.time_s,
                "upper_current_density_a_m2": upper.current_density_a_m2,
                "upper_voltage_v": upper.cell_voltage_v,
                "voltage_hold_count": len(self.voltage_sweep.holds),
                "sweep_end_s": self.voltage_sweep.holds[-1].end_s,
                "solver_clock_step": self.voltage_sweep.delta_t_s,
                "solver_outer_iterations": self.voltage_sweep.outer_iterations,
                "duration_s": self.duration_s,
                "solver_log": str(self.solver_log),
                "mesh_log": str(self.mesh_log),
            },
        )


def _format_number(value: float) -> str:
    return f"{0.0 if abs(value) < 5.0e-14 else value:.12g}"


def _vertices_span(text: str) -> tuple[int, int]:
    match = re.search(r"\bvertices\s*\(\s*", text)
    if not match:
        raise OptimizationError("Cannot find a vertices block in blockMeshDict")
    end = text.find("\n);", match.end())
    if end == -1:
        raise OptimizationError("Cannot find the end of the vertices block")
    return match.end(), end


def _unique_sorted(values: Iterable[float], tolerance: float = 1.0e-10) -> list[float]:
    result: list[float] = []
    for value in sorted(values):
        if not result or not math.isclose(value, result[-1], abs_tol=tolerance):
            result.append(value)
    return result


def rewrite_block_mesh_thickness(block_mesh_path: Path, thickness_um: float) -> GeometryUpdate:
    """Set centered membrane thickness while preserving all other layer widths."""
    if not math.isfinite(thickness_um) or thickness_um <= 0:
        raise OptimizationError("Membrane thickness must be a finite positive value")
    text = block_mesh_path.read_text(encoding="utf-8")
    start, end = _vertices_span(text)
    vertex_text = text[start:end]
    matches = list(VERTEX_RE.finditer(vertex_text))
    if not matches:
        raise OptimizationError("No vertices found in blockMeshDict")
    z_levels = _unique_sorted(float(match.group(3)) for match in matches)
    negative, positive = [z for z in z_levels if z < 0], [z for z in z_levels if z > 0]
    if not negative or not positive:
        raise OptimizationError("Membrane rewrite expects planes on both sides of z=0")
    lower, upper = max(negative), min(positive)
    center = 0.5 * (lower + upper)
    old_thickness = upper - lower
    new_thickness = thickness_um / 1000.0
    shift = 0.5 * (new_thickness - old_thickness)

    old_spacings = [
        z_levels[index + 1] - z_levels[index]
        for index in range(len(z_levels) - 1)
        if not (math.isclose(z_levels[index], lower) and math.isclose(z_levels[index + 1], upper))
    ]

    def replace(match: re.Match[str]) -> str:
        x, y, z = (float(match.group(index)) for index in range(1, 4))
        if z > center:
            z += shift
        elif z < center:
            z -= shift
        else:
            raise OptimizationError("A vertex lies on the membrane center plane")
        return f"({_format_number(x)} {_format_number(y)} {_format_number(z)})"

    rewritten_vertices = VERTEX_RE.sub(replace, vertex_text)
    new_z = _unique_sorted(float(match.group(3)) for match in VERTEX_RE.finditer(rewritten_vertices))
    new_negative, new_positive = [z for z in new_z if z < center], [z for z in new_z if z > center]
    actual = min(new_positive) - max(new_negative)
    if not math.isclose(actual, new_thickness, abs_tol=1.0e-10):
        raise OptimizationError("Membrane rewrite verification failed")
    new_spacings = [
        new_z[index + 1] - new_z[index]
        for index in range(len(new_z) - 1)
        if index != len(new_negative) - 1
    ]
    if len(old_spacings) != len(new_spacings) or any(
        not math.isclose(before, after, abs_tol=1.0e-10)
        for before, after in zip(old_spacings, new_spacings)
    ):
        raise OptimizationError("A non-membrane layer thickness changed unexpectedly")
    block_mesh_path.write_text(text[:start] + rewritten_vertices + text[end:], encoding="utf-8")
    return GeometryUpdate(old_thickness * 1000.0, thickness_um, center, shift, len(matches))


def _matching_delimiter(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    raise OptimizationError(f"Unmatched {opening!r} in OpenFOAM dictionary")


def _named_block_span(text: str, name: str) -> tuple[int, int]:
    match = re.search(rf"\b{re.escape(name)}\b\s*\{{", text)
    if not match:
        raise OptimizationError(f"Cannot find {name!r} dictionary block")
    opening = text.find("{", match.start())
    return opening + 1, _matching_delimiter(text, opening, "{", "}")


def _replace_control_scalar(path: Path, keyword: str, value: float, required: bool = True) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    rewritten, count = re.subn(
        rf"(?m)^(\s*{re.escape(keyword)}\s+)({FLOAT_PATTERN})(\s*;)",
        lambda match: match.group(1) + _format_number(value) + match.group(3),
        text,
        count=1,
    )
    if count != 1 and required:
        raise OptimizationError(f"Cannot update {keyword} in: {path}")
    if count:
        path.write_text(rewritten, encoding="utf-8")


def _set_polarization_curve_active(text: str, active: bool) -> str:
    """Set the optional convergence-controlled current-scan switch."""
    if not re.search(r"\bpolarizationCurve\b\s*\{", text):
        return text
    start, end = _named_block_span(text, "polarizationCurve")
    block = text[start:end]
    replacement = "true" if active else "false"
    rewritten, count = re.subn(
        r"(?m)^(\s*active\s+)(?:true|false)(\s*;)",
        lambda match: match.group(1) + replacement + match.group(2),
        block,
        count=1,
    )
    if count != 1:
        raise OptimizationError("Cannot update polarizationCurve active switch")
    return text[:start] + rewritten + text[end:]


def _set_galvanostatic_active(text: str, active: bool) -> str:
    """Select current or imposed-voltage control in a scratch case."""
    start, end = _named_block_span(text, "galvanostatic")
    block = text[start:end]
    replacement = "true" if active else "false"
    rewritten, count = re.subn(
        r"(?m)^(\s*active\s+)(?:true|false)(\s*;)",
        lambda match: match.group(1) + replacement + match.group(2),
        block,
        count=1,
    )
    if count != 1:
        raise OptimizationError("Cannot update galvanostatic active switch")
    return text[:start] + rewritten + text[end:]


def configure_voltage_sweep(
    case_path: Path, config: AemecEvaluationConfig
) -> VoltageSweep:
    """Retain the case voltage table and configure a complete sweep run."""
    config.validate()
    region_path = case_path / "constant/phiEAnode/regionProperties"
    text = region_path.read_text(encoding="utf-8")
    text = _set_galvanostatic_active(text, False)
    text = _set_polarization_curve_active(text, False)

    gs_start, gs_end = _named_block_span(text, "galvanostatic")
    voltage_start_rel, voltage_end_rel = _named_block_span(
        text[gs_start:gs_end], "voltage"
    )
    voltage_start = gs_start + voltage_start_rel
    voltage_end = gs_start + voltage_end_rel
    voltage_text = text[voltage_start:voltage_end]
    if not re.search(r"(?m)^\s*type\s+table\s*;", voltage_text):
        raise OptimizationError("Optimization requires a voltage table")
    values_match = re.search(r"\bvalues\s*\(", voltage_text)
    if not values_match:
        raise OptimizationError("Cannot find voltage-table values")
    values_open_rel = voltage_text.find("(", values_match.start())
    values_close_rel = _matching_delimiter(
        voltage_text, values_open_rel, "(", ")"
    )
    pairs = [
        (float(match.group(1)), float(match.group(2)))
        for match in TABLE_PAIR_RE.finditer(
            voltage_text[values_open_rel + 1:values_close_rel]
        )
    ]
    if len(pairs) < 2 or len(pairs) % 2:
        raise OptimizationError(
            "Voltage table must contain start/end pairs for every hold"
        )

    holds: list[VoltageHold] = []
    for index in range(0, len(pairs), 2):
        (start_s, start_voltage), (end_s, end_voltage) = pairs[index:index + 2]
        if end_s <= start_s:
            raise OptimizationError("Every voltage hold must have positive duration")
        if not math.isclose(
            start_voltage, end_voltage, rel_tol=0.0, abs_tol=1.0e-10
        ):
            raise OptimizationError(
                "Every voltage hold must repeat the same voltage at its start and end"
            )
        if holds and start_s <= holds[-1].end_s:
            raise OptimizationError("Voltage holds must have increasing time ranges")
        holds.append(VoltageHold(start_s, end_s, start_voltage))

    if any(
        current.voltage_v <= previous.voltage_v
        for previous, current in zip(holds, holds[1:])
    ):
        raise OptimizationError("Voltage sweep values must increase strictly")

    control_run = case_path / "system/controlDict.run"
    control_text = control_run.read_text(encoding="utf-8")
    delta_match = re.search(
        rf"(?m)^\s*deltaT\s+({FLOAT_PATTERN})\s*;", control_text
    )
    if not delta_match:
        raise OptimizationError("Cannot read deltaT from controlDict.run")
    delta_t = float(delta_match.group(1))
    if delta_t <= 0 or not math.isfinite(delta_t):
        raise OptimizationError("OpenFOAM deltaT must be positive")
    if any(
        (hold.end_s - hold.start_s) / delta_t + 1.0e-9
        < config.stability_samples
        for hold in holds
    ):
        raise OptimizationError("A voltage hold is shorter than the stability window")

    end_time = holds[-1].end_s
    outer_iterations = int(math.ceil(end_time / delta_t - 1.0e-12))
    region_path.write_text(text, encoding="utf-8")
    for control_path in (control_run, case_path / "system/controlDict"):
        _replace_control_scalar(control_path, "endTime", end_time)
        _replace_control_scalar(control_path, "writeInterval", end_time)
        _replace_control_scalar(control_path, "purgeWrite", 1, required=False)
    return VoltageSweep(tuple(holds), delta_t, outer_iterations)


def parse_voltage_sweep_samples(log_text: str) -> list[SweepSample]:
    """Pair post-solve collector current and crossover records by time step."""
    current_time = current = voltage = None
    samples: list[SweepSample] = []
    for line in log_text.splitlines():
        match = TIME_RE.search(line)
        if match:
            current_time = float(match.group(1))
            current = voltage = None
            continue
        match = BOUNDARY_POINT_RE.search(line)
        if match:
            current, voltage = float(match.group(1)), float(match.group(2))
            continue
        match = CROSSOVER_RE.search(line)
        if match:
            crossover = float(match.group(1))
            values = (current_time, current, voltage, crossover)
            if all(value is not None and math.isfinite(value) for value in values):
                samples.append(
                    SweepSample(
                        float(current_time),
                        float(current),
                        float(voltage),
                        crossover,
                    )
                )
    return samples


def _voltage_hold_window(
    samples: Sequence[SweepSample],
    hold: VoltageHold,
    sweep: VoltageSweep,
    config: AemecEvaluationConfig,
) -> list[SweepSample]:
    time_tolerance = max(1.0e-8, 0.25 * sweep.delta_t_s)
    voltage_tolerance = max(1.0e-8, config.voltage_stability_tolerance_v)
    hold_samples = [
        sample
        for sample in samples
        if hold.start_s - time_tolerance < sample.time_s <= hold.end_s + time_tolerance
        and math.isclose(
            sample.cell_voltage_v,
            hold.voltage_v,
            rel_tol=0.0,
            abs_tol=voltage_tolerance,
        )
    ]
    if len(hold_samples) < config.stability_samples:
        raise OptimizationError(
            f"Not enough paired samples at the {hold.voltage_v:g} V hold"
        )
    window = hold_samples[-config.stability_samples:]
    if not math.isclose(
        window[-1].time_s,
        hold.end_s,
        rel_tol=0.0,
        abs_tol=time_tolerance,
    ):
        raise OptimizationError(
            f"The final sample is missing from the {hold.voltage_v:g} V hold"
        )
    for previous, current_sample in zip(window, window[1:]):
        if not math.isclose(
            current_sample.time_s - previous.time_s,
            sweep.delta_t_s,
            rel_tol=0.0,
            abs_tol=time_tolerance,
        ):
            raise OptimizationError(
                f"Final samples are not consecutive at {hold.voltage_v:g} V"
            )
    if any(
        sample.cell_voltage_v <= 0 or sample.crossover_rate_mol_s < 0
        for sample in window
    ):
        raise OptimizationError(
            f"The {hold.voltage_v:g} V hold contains non-physical objectives"
        )
    return window


def _validate_voltage_hold_stability(
    window: Sequence[SweepSample],
    hold: VoltageHold,
    config: AemecEvaluationConfig,
) -> None:
    """Validate only a hold that contributes to the interpolated objective."""
    voltage_values = [sample.cell_voltage_v for sample in window]
    if (
        max(voltage_values) - min(voltage_values)
        > config.voltage_stability_tolerance_v
    ):
        raise OptimizationError(
            f"Cell voltage is not stable at the {hold.voltage_v:g} V hold"
        )
    current_values = [abs(sample.current_density_a_m2) for sample in window]
    current_mean = sum(current_values) / len(current_values)
    current_scale = max(
        current_mean, 0.01 * config.target_current_density_a_m2, 1.0e-30
    )
    current_relative_std_dev = math.sqrt(
        sum((value - current_mean) ** 2 for value in current_values)
        / len(current_values)
    ) / current_scale
    if current_relative_std_dev > config.current_relative_tolerance:
        raise OptimizationError(
            f"Collector current is not stable at the {hold.voltage_v:g} V hold: "
            f"relative standard deviation={current_relative_std_dev:.6g}, "
            f"limit={config.current_relative_tolerance:.6g}"
        )
    crossover_values = [sample.crossover_rate_mol_s for sample in window]
    crossover_scale = max(
        max(abs(value) for value in crossover_values), 1.0e-30
    )
    crossover_relative_range = (
        max(crossover_values) - min(crossover_values)
    ) / crossover_scale
    if crossover_relative_range > config.crossover_stability_relative_tolerance:
        raise OptimizationError(
            f"Hydrogen crossover is not stable at the {hold.voltage_v:g} V hold: "
            f"relative range={crossover_relative_range:.6g}, "
            f"limit={config.crossover_stability_relative_tolerance:.6g}"
        )


def interpolate_voltage_objective(
    log_text: str,
    config: AemecEvaluationConfig,
    sweep: VoltageSweep,
) -> InterpolatedObjective:
    """Linearly interpolate voltage and crossover at the requested current."""
    if not NORMAL_END_RE.search(log_text):
        raise OptimizationError(
            "Solver log does not contain the normal OpenFOAM 'End'"
        )
    fatal = FATAL_OUTPUT_RE.search(log_text)
    if fatal:
        raise OptimizationError(
            f"Solver log contains a fatal marker: {fatal.group(0)!r}"
        )
    samples = parse_voltage_sweep_samples(log_text)
    # Every hold must have complete paired output, but stability is relevant
    # only for the one exact point or two points used by the interpolation.
    # Applying a relative crossover test to an unused, near-zero 1.3 V point
    # makes the full trial fail on numerical noise rather than objective quality.
    windows = [
        _voltage_hold_window(samples, hold, sweep, config)
        for hold in sweep.holds
    ]
    points = [window[-1] for window in windows]
    target = config.target_current_density_a_m2
    exact_indices = [
        index
        for index, point in enumerate(points)
        if math.isclose(
            abs(point.current_density_a_m2),
            target,
            rel_tol=1.0e-12,
            abs_tol=1.0e-6,
        )
    ]
    if len(exact_indices) > 1:
        raise OptimizationError(
            "The voltage sweep contains duplicate target-current points"
        )
    if exact_indices:
        index = exact_indices[0]
        point = points[index]
        _validate_voltage_hold_stability(
            windows[index], sweep.holds[index], config
        )
        return InterpolatedObjective(
            target,
            point.cell_voltage_v,
            point.crossover_rate_mol_s,
            0.0,
            point,
            point,
        )

    brackets: list[tuple[int, int]] = []
    for index, (first, second) in enumerate(zip(points, points[1:])):
        first_current = abs(first.current_density_a_m2)
        second_current = abs(second.current_density_a_m2)
        if (first_current - target) * (second_current - target) < 0:
            brackets.append((index, index + 1))
    if not brackets:
        current_min = min(abs(point.current_density_a_m2) for point in points)
        current_max = max(abs(point.current_density_a_m2) for point in points)
        raise OptimizationError(
            "Voltage sweep does not bracket the target current density: "
            f"target={target:g} A/m2, sampled range={current_min:g}..{current_max:g} A/m2"
        )
    if len(brackets) > 1:
        raise OptimizationError(
            "The polarization curve crosses the target current more than once"
        )

    first_index, second_index = brackets[0]
    for index in (first_index, second_index):
        _validate_voltage_hold_stability(
            windows[index], sweep.holds[index], config
        )
    first, second = points[first_index], points[second_index]
    if abs(first.current_density_a_m2) <= abs(second.current_density_a_m2):
        lower, upper = first, second
    else:
        lower, upper = second, first
    lower_current = abs(lower.current_density_a_m2)
    upper_current = abs(upper.current_density_a_m2)
    fraction = (target - lower_current) / (upper_current - lower_current)
    voltage = lower.cell_voltage_v + fraction * (
        upper.cell_voltage_v - lower.cell_voltage_v
    )
    crossover = lower.crossover_rate_mol_s + fraction * (
        upper.crossover_rate_mol_s - lower.crossover_rate_mol_s
    )
    return InterpolatedObjective(
        target,
        voltage,
        crossover,
        fraction,
        lower,
        upper,
    )


def _case_copy_ignore(_: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in {".DS_Store", "VTK", "postProcessing", "dynamicCode", "polyMesh", "salome.unv"}:
            ignored.add(name)
        elif name.startswith("log.") or name.startswith("processor"):
            ignored.add(name)
        elif name != "0.orig" and re.fullmatch(FLOAT_PATTERN, name):
            ignored.add(name)
    return ignored


def _paths_overlap(first: Path, second: Path) -> bool:
    first, second = first.resolve(), second.resolve()
    return first == second or first in second.parents or second in first.parents


def validate_path_layout(source_case: Path, work_case: Path, output_dir: Path) -> None:
    paths = {"source case": source_case.resolve(), "scratch case": work_case.resolve(), "output directory": output_dir.resolve()}
    for first, second in (("source case", "scratch case"), ("source case", "output directory"), ("scratch case", "output directory")):
        if _paths_overlap(paths[first], paths[second]):
            raise OptimizationError(f"Unsafe overlapping paths: {first}={paths[first]} and {second}={paths[second]}")


def case_fingerprint(source_case: Path) -> str:
    if not source_case.is_dir():
        raise OptimizationError(f"OpenFOAM case directory not found: {source_case}")
    digest = hashlib.sha256()
    for path in sorted(source_case.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source_case)
        if any(part in {".DS_Store", "VTK", "postProcessing", "dynamicCode", "polyMesh"} or part.startswith(("log.", "processor")) or (part != "0.orig" and re.fullmatch(FLOAT_PATTERN, part)) for part in relative.parts):
            continue
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def copy_clean_case(source_case: Path, destination_case: Path) -> None:
    source_case, destination_case = source_case.resolve(), destination_case.resolve()
    if not source_case.is_dir():
        raise OptimizationError(f"OpenFOAM case directory not found: {source_case}")
    if _paths_overlap(source_case, destination_case):
        raise OptimizationError("Source and scratch cases must not overlap")
    if destination_case.exists():
        shutil.rmtree(destination_case)
    destination_case.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_case, destination_case, ignore=_case_copy_ignore)


def _terminate_process_group(process: subprocess.Popen[str]) -> str:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        output, _ = process.communicate(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output, _ = process.communicate()
    return output


def run_command(command: Sequence[str], cwd: Path, log_path: Path, timeout_s: float | None) -> str:
    if not command:
        raise OptimizationError("An empty external command was provided")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    display = shlex.join(command)
    started = time.monotonic()
    try:
        process = subprocess.Popen(list(command), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
    except FileNotFoundError as exc:
        log_path.write_text(f"Command not found: {command[0]}\n", encoding="utf-8")
        raise OptimizationError(f"Cannot run {command[0]!r}; source the OpenFOAM environment first") from exc
    try:
        output, _ = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        output = _terminate_process_group(process)
        log_path.write_text(output, encoding="utf-8")
        raise OptimizationError(f"Command timed out after {timeout_s:g} s: {display}") from exc
    except KeyboardInterrupt:
        output = _terminate_process_group(process)
        log_path.write_text(f"$ {display}\n# interrupted by user\n{output}", encoding="utf-8")
        raise
    log_path.write_text(f"$ {display}\n# elapsed_s={time.monotonic() - started:.6f}\n{output}", encoding="utf-8")
    if process.returncode != 0:
        raise OptimizationError(f"Command exited with status {process.returncode}: {display}. See {log_path}")
    fatal = FATAL_OUTPUT_RE.search(output)
    if fatal:
        raise OptimizationError(f"Command output contains {fatal.group(0)!r}: {display}. See {log_path}")
    return output


class AemecOpenFoamEvaluator:
    """Evaluate a membrane thickness in a fresh OpenFOAM scratch case."""
    required_mesh_regions = (
        "anode",
        "cathode",
        "electrolyte",
        "interconnect",
        "phiECathode",
        "phiEAnode",
        "phiAnion",
    )

    def __init__(self, source_case: Path, work_case: Path, logs_dir: Path, mesh_command: Sequence[str], solver_command: Sequence[str], config: AemecEvaluationConfig, timeout_s: float | None) -> None:
        self.source_case, self.work_case, self.logs_dir = source_case.resolve(), work_case.resolve(), logs_dir.resolve()
        self.mesh_command, self.solver_command = tuple(mesh_command), tuple(solver_command)
        self.config, self.timeout_s = config, timeout_s
        config.validate()

    def _verify_mesh(self) -> None:
        expected = [self.work_case / "constant/polyMesh/points"] + [self.work_case / "constant" / region / "polyMesh/points" for region in self.required_mesh_regions]
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise OptimizationError("Mesh command completed but required mesh files are missing:\n  " + "\n  ".join(missing))

    def evaluate(self, trial_number: int, thickness_um: float) -> AemecEvaluation:
        started = time.monotonic()
        copy_clean_case(self.source_case, self.work_case)
        rewrite_block_mesh_thickness(self.work_case / "system/blockMeshDict", thickness_um)
        sweep = configure_voltage_sweep(self.work_case, self.config)
        mesh_log = self.logs_dir / f"trial_{trial_number:04d}_mesh.log"
        solver_log = self.logs_dir / f"trial_{trial_number:04d}_solver.log"
        run_command(self.mesh_command, self.work_case, mesh_log, self.timeout_s)
        self._verify_mesh()
        solver_output = run_command(self.solver_command, self.work_case, solver_log, self.timeout_s)
        objective = interpolate_voltage_objective(
            solver_output, self.config, sweep
        )
        return AemecEvaluation(
            objective,
            sweep,
            solver_log,
            mesh_log,
            time.monotonic() - started,
        )
