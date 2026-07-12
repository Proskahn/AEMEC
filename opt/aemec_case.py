"""AEMEC/OpenFOAM adapter for the generic optimization engine.

This module contains the case-specific science and runtime integration: the
membrane geometry convention, galvanostatic schedule, log contract, and safe
OpenFOAM scratch-case execution.  It does not know how Optuna selects designs
or how Pareto reports are rendered.
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


# Version 5 records the post-solve boundary-current feedback controller. It
# must not resume studies made with the old pre-solve reaction-source feedback.
OBJECTIVE_SCHEMA_VERSION = 5
DEFAULT_TARGET_CURRENT_DENSITY_A_M2 = 10_000.0
DEFAULT_CURRENT_RELATIVE_TOLERANCE = 0.05
DEFAULT_TARGET_HOLD_DURATION_S = 30.0
DEFAULT_RUN_MODE = "fast"
DEFAULT_SOLVER_ITERATIONS = 250
DEFAULT_ITERATION_CLOCK_STEP = 1.0
DEFAULT_STABILITY_SAMPLES = 5
DEFAULT_VOLTAGE_STABILITY_TOLERANCE_V = 0.005
DEFAULT_CROSSOVER_STABILITY_RELATIVE_TOLERANCE = 0.02
DEFAULT_CONTROLLER_STEP_TOLERANCE_V = 0.001

FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
VERTEX_RE = re.compile(
    rf"\(\s*({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s*\)"
)
TABLE_PAIR_RE = re.compile(rf"\(\s*({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s*\)")
TIME_RE = re.compile(rf"^\s*Time\s*=\s*({FLOAT_PATTERN})\s*$")
TARGET_CURRENT_RE = re.compile(
    rf"\bgalvanostatic\s+target:\s*({FLOAT_PATTERN})\s*A/m2\b", re.IGNORECASE
)
CONTROLLER_STEP_RE = re.compile(
    rf"\bgalvanostatic\s+target:.*?\blimited\s+dV:\s*({FLOAT_PATTERN})\b",
    re.IGNORECASE,
)
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
    run_mode: str = DEFAULT_RUN_MODE
    solver_iterations: int = DEFAULT_SOLVER_ITERATIONS
    iteration_clock_step: float = DEFAULT_ITERATION_CLOCK_STEP
    target_hold_duration_s: float = DEFAULT_TARGET_HOLD_DURATION_S
    stability_samples: int = DEFAULT_STABILITY_SAMPLES
    voltage_stability_tolerance_v: float = DEFAULT_VOLTAGE_STABILITY_TOLERANCE_V
    crossover_stability_relative_tolerance: float = (
        DEFAULT_CROSSOVER_STABILITY_RELATIVE_TOLERANCE
    )
    controller_step_tolerance_v: float = DEFAULT_CONTROLLER_STEP_TOLERANCE_V

    def validate(self) -> None:
        if (
            not math.isfinite(self.target_current_density_a_m2)
            or self.target_current_density_a_m2 <= 0
        ):
            raise OptimizationError("Target current-density magnitude must be positive")
        if not 0 < self.current_relative_tolerance < 1:
            raise OptimizationError("Current relative tolerance must be in (0, 1)")
        if self.run_mode not in {"fast", "ramp"}:
            raise OptimizationError("Run mode must be 'fast' or 'ramp'")
        if self.run_mode == "fast" and self.solver_iterations < self.stability_samples:
            raise OptimizationError("Fast-mode iteration budget is shorter than stability window")
        if self.run_mode == "fast" and (
            not math.isfinite(self.iteration_clock_step)
            or self.iteration_clock_step <= 0
        ):
            raise OptimizationError("Iteration-clock step must be positive")
        if self.run_mode == "ramp" and (
            not math.isfinite(self.target_hold_duration_s)
            or self.target_hold_duration_s <= 0
        ):
            raise OptimizationError("Ramp-mode target hold duration must be positive")
        if self.stability_samples <= 0:
            raise OptimizationError("Stability sample count must be positive")
        if (
            not math.isfinite(self.voltage_stability_tolerance_v)
            or self.voltage_stability_tolerance_v < 0
        ):
            raise OptimizationError("Voltage stability tolerance cannot be negative")
        if not 0 <= self.crossover_stability_relative_tolerance < 1:
            raise OptimizationError("Crossover stability tolerance must be in [0, 1)")
        if (
            not math.isfinite(self.controller_step_tolerance_v)
            or self.controller_step_tolerance_v < 0
        ):
            raise OptimizationError("Controller-step tolerance cannot be negative")

    def resume_settings(self) -> dict[str, object]:
        self.validate()
        return {
            "objective_schema_version": OBJECTIVE_SCHEMA_VERSION,
            "target_current_density_a_m2": self.target_current_density_a_m2,
            "current_relative_tolerance": self.current_relative_tolerance,
            "run_mode": self.run_mode,
            "solver_iterations": self.solver_iterations if self.run_mode == "fast" else None,
            "iteration_clock_step": self.iteration_clock_step if self.run_mode == "fast" else None,
            "target_hold_duration_s": (
                self.target_hold_duration_s if self.run_mode == "ramp" else None
            ),
            "stability_samples": self.stability_samples,
            "voltage_stability_tolerance_v": self.voltage_stability_tolerance_v,
            "crossover_stability_relative_tolerance": self.crossover_stability_relative_tolerance,
            "controller_step_tolerance_v": self.controller_step_tolerance_v,
        }


@dataclass(frozen=True)
class GeometryUpdate:
    old_thickness_um: float
    new_thickness_um: float
    center_mm: float
    half_stack_shift_mm: float
    vertex_count: int


@dataclass(frozen=True)
class TargetHold:
    start_s: float
    end_s: float
    delta_t_s: float
    outer_iterations: int
    signed_current_density_a_m2: float


@dataclass(frozen=True)
class ObjectiveSample:
    time_s: float
    target_current_density_a_m2: float
    actual_current_density_a_m2: float
    cell_voltage_v: float
    crossover_rate_mol_s: float
    controller_voltage_step_v: float


@dataclass(frozen=True)
class AemecEvaluation:
    sample: ObjectiveSample
    target_hold: TargetHold
    solver_log: Path
    mesh_log: Path
    duration_s: float

    def as_objective_result(self) -> ObjectiveResult:
        return ObjectiveResult(
            values=(self.sample.cell_voltage_v, self.sample.crossover_rate_mol_s),
            metadata={
                "sample_time_s": self.sample.time_s,
                "actual_current_density_a_m2": self.sample.actual_current_density_a_m2,
                "controller_voltage_step_v": self.sample.controller_voltage_step_v,
                "target_hold_start_s": self.target_hold.start_s,
                "target_hold_end_s": self.target_hold.end_s,
                "solver_clock_step": self.target_hold.delta_t_s,
                "solver_outer_iterations": self.target_hold.outer_iterations,
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
    """Disable the multi-target stable scan in an optimizer scratch case.

    Optimization owns one direct current target per CFD evaluation; it retains
    its separate final-window stability validation below.
    """
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


def configure_target_hold(case_path: Path, config: AemecEvaluationConfig) -> TargetHold:
    """Configure either a direct-target fast run or the original ramped run."""
    config.validate()
    # Galvanostatic control is applied at the physical oxygen/anode collector.
    region_path = case_path / "constant/phiEAnode/regionProperties"
    text = region_path.read_text(encoding="utf-8")
    text = _set_polarization_curve_active(text, False)
    gs_start, gs_end = _named_block_span(text, "galvanostatic")
    ibar_start_rel, ibar_end_rel = _named_block_span(text[gs_start:gs_end], "ibar")
    ibar_start, ibar_end = gs_start + ibar_start_rel, gs_start + ibar_end_rel
    ibar_text = text[ibar_start:ibar_end]
    values_match = re.search(r"\bvalues\s*\(", ibar_text)
    if not values_match:
        raise OptimizationError("Cannot find galvanostatic ibar values")
    values_open_rel = ibar_text.find("(", values_match.start())
    values_close_rel = _matching_delimiter(ibar_text, values_open_rel, "(", ")")
    values_open, values_close = ibar_start + values_open_rel, ibar_start + values_close_rel
    pairs = [
        (float(match.group(1)), float(match.group(2)))
        for match in TABLE_PAIR_RE.finditer(text[values_open + 1:values_close])
    ]
    target_index = next(
        (
            index for index, (_, current) in enumerate(pairs)
            if math.isclose(abs(current), config.target_current_density_a_m2, rel_tol=1.0e-9, abs_tol=1.0e-6)
        ),
        None,
    )
    if target_index is None:
        raise OptimizationError("The galvanostatic table has no requested current hold")
    control_run = case_path / "system/controlDict.run"
    control_text = control_run.read_text(encoding="utf-8")
    delta_match = re.search(rf"(?m)^\s*deltaT\s+({FLOAT_PATTERN})\s*;", control_text)
    if not delta_match:
        raise OptimizationError("Cannot read deltaT from controlDict.run")
    original_delta = float(delta_match.group(1))
    if original_delta <= 0 or not math.isfinite(original_delta):
        raise OptimizationError("OpenFOAM deltaT must be positive")
    target_time, signed_target = pairs[target_index]
    if config.run_mode == "fast":
        delta_t = config.iteration_clock_step
        first_time = delta_t
        end_time = config.solver_iterations * delta_t
        outer_iterations = config.solver_iterations
        available = config.solver_iterations
        rewritten_pairs = [(0.0, signed_target), (end_time, signed_target)]
    else:
        delta_t = original_delta
        first_time = math.ceil((target_time - 1.0e-12) / delta_t) * delta_t
        end_time = math.ceil((first_time + config.target_hold_duration_s - 1.0e-12) / delta_t) * delta_t
        outer_iterations = int(round(end_time / delta_t))
        available = int(math.floor((end_time - first_time) / delta_t + 1.0e-9)) + 1
        rewritten_pairs = pairs[:target_index + 1] + [(end_time, signed_target)]
    if available < config.stability_samples:
        raise OptimizationError("Target hold is shorter than the stability window")
    pair_lines = "\n".join(
        f"            ({_format_number(moment)}  {_format_number(current)})"
        for moment, current in rewritten_pairs
    )
    region_path.write_text(
        text[:values_open + 1] + "\n" + pair_lines + "\n        " + text[values_close:],
        encoding="utf-8",
    )
    for control_path in (control_run, case_path / "system/controlDict"):
        _replace_control_scalar(control_path, "endTime", end_time)
        if config.run_mode == "fast":
            _replace_control_scalar(control_path, "deltaT", delta_t)
        _replace_control_scalar(control_path, "writeInterval", end_time)
        _replace_control_scalar(control_path, "purgeWrite", 1, required=False)
    return TargetHold(first_time, end_time, delta_t, outer_iterations, signed_target)


def parse_objective_samples(log_text: str) -> list[ObjectiveSample]:
    current_time = target = current = voltage = controller_step = None
    samples: list[ObjectiveSample] = []
    for line in log_text.splitlines():
        match = TIME_RE.search(line)
        if match:
            current_time = float(match.group(1))
            target = current = voltage = controller_step = None
            continue
        match = TARGET_CURRENT_RE.search(line)
        if match:
            target = float(match.group(1))
        match = CONTROLLER_STEP_RE.search(line)
        if match:
            controller_step = float(match.group(1))
        match = BOUNDARY_POINT_RE.search(line)
        if match:
            current, voltage = float(match.group(1)), float(match.group(2))
            continue
        match = CROSSOVER_RE.search(line)
        if match:
            crossover = float(match.group(1))
            values = (current_time, target, current, voltage, controller_step, crossover)
            if all(value is not None and math.isfinite(value) for value in values):
                samples.append(ObjectiveSample(float(current_time), float(target), float(current), float(voltage), crossover, float(controller_step)))
    return samples


def select_target_sample(log_text: str, config: AemecEvaluationConfig, hold: TargetHold) -> ObjectiveSample:
    """Require a complete, consecutive, stable final window at the signed target."""
    if not NORMAL_END_RE.search(log_text):
        raise OptimizationError("Solver log does not contain the normal OpenFOAM 'End'")
    fatal = FATAL_OUTPUT_RE.search(log_text)
    if fatal:
        raise OptimizationError(f"Solver log contains a fatal marker: {fatal.group(0)!r}")
    samples = [
        sample for sample in parse_objective_samples(log_text)
        if math.isclose(abs(sample.target_current_density_a_m2), config.target_current_density_a_m2, rel_tol=1.0e-9, abs_tol=1.0e-6)
    ]
    if len(samples) < config.stability_samples:
        raise OptimizationError("Not enough paired target-current objective samples")
    window = samples[-config.stability_samples:]
    time_tolerance = max(1.0e-8, 0.25 * hold.delta_t_s)
    if not math.isclose(window[-1].time_s, hold.end_s, rel_tol=0, abs_tol=time_tolerance):
        raise OptimizationError("The final objective sample is missing")
    for previous, current in zip(window, window[1:]):
        if not math.isclose(current.time_s - previous.time_s, hold.delta_t_s, rel_tol=0, abs_tol=time_tolerance):
            raise OptimizationError("Final objective samples are not consecutive")
    allowed_current_error = config.current_relative_tolerance * config.target_current_density_a_m2
    for sample in window:
        if abs(sample.actual_current_density_a_m2 - sample.target_current_density_a_m2) > allowed_current_error:
            raise OptimizationError("The final target-current window is outside tolerance")
        if sample.cell_voltage_v <= 0 or sample.crossover_rate_mol_s < 0:
            raise OptimizationError("Final target-current window contains non-physical objectives")
    if max(abs(sample.controller_voltage_step_v) for sample in window) > config.controller_step_tolerance_v:
        raise OptimizationError("The galvanostatic controller is still moving")
    voltage_range = max(sample.cell_voltage_v for sample in window) - min(sample.cell_voltage_v for sample in window)
    if voltage_range > config.voltage_stability_tolerance_v:
        raise OptimizationError("Cell voltage is not stable")
    crossover_values = [sample.crossover_rate_mol_s for sample in window]
    crossover_range = max(crossover_values) - min(crossover_values)
    scale = max(max(abs(value) for value in crossover_values), 1.0e-30)
    if crossover_range / scale > config.crossover_stability_relative_tolerance:
        raise OptimizationError("Hydrogen crossover is not stable")
    return window[-1]


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
        hold = configure_target_hold(self.work_case, self.config)
        mesh_log = self.logs_dir / f"trial_{trial_number:04d}_mesh.log"
        solver_log = self.logs_dir / f"trial_{trial_number:04d}_solver.log"
        run_command(self.mesh_command, self.work_case, mesh_log, self.timeout_s)
        self._verify_mesh()
        solver_output = run_command(self.solver_command, self.work_case, solver_log, self.timeout_s)
        sample = select_target_sample(solver_output, self.config, hold)
        return AemecEvaluation(sample, hold, solver_log, mesh_log, time.monotonic() - started)
