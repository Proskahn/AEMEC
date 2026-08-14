"""Experiment definition and deterministic OpenFOAM dictionary updates."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .project import OptimizationError, rewrite_block_mesh_thickness


FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
DEFAULT_THICKNESSES_UM = (20.0, 40.0, 60.0, 80.0)


@dataclass(frozen=True)
class ExperimentConfig:
    """Scientific settings for the fixed-current thickness comparison."""

    thicknesses_um: tuple[float, ...] = DEFAULT_THICKNESSES_UM
    target_current_density_a_m2: float = 10_000.0
    duration_s: float = 20.0
    delta_t_s: float = 0.1
    initial_voltage_v: float = 2.0
    target_current_tolerance: float = 0.01
    current_stability_tolerance: float = 0.01
    voltage_tolerance_v: float = 0.002
    stability_samples: int = 5
    final_window_s: float = 5.0

    @property
    def signed_target_current_density_a_m2(self) -> float:
        """Electrolysis is negative in the AEMEC solver sign convention."""
        return -self.target_current_density_a_m2

    def validate(self) -> None:
        values = tuple(float(value) for value in self.thicknesses_um)
        if not values or any(not math.isfinite(value) or value <= 0 for value in values):
            raise OptimizationError("Membrane thicknesses must be finite positive values")
        if len(set(values)) != len(values):
            raise OptimizationError("Membrane thicknesses must be unique")
        if not math.isfinite(self.target_current_density_a_m2) or self.target_current_density_a_m2 <= 0:
            raise OptimizationError("Target current-density magnitude must be positive")
        for name, value in (
            ("duration", self.duration_s),
            ("time step", self.delta_t_s),
            ("initial voltage", self.initial_voltage_v),
            ("final averaging window", self.final_window_s),
        ):
            if not math.isfinite(value) or value <= 0:
                raise OptimizationError(f"Experiment {name} must be positive")
        if self.delta_t_s > self.duration_s:
            raise OptimizationError("The time step cannot exceed the experiment duration")
        if self.final_window_s > self.duration_s:
            raise OptimizationError("The final averaging window cannot exceed the duration")
        for name, value in (
            ("target-current tolerance", self.target_current_tolerance),
            ("current-stability tolerance", self.current_stability_tolerance),
        ):
            if not 0 < value < 1:
                raise OptimizationError(f"The {name} must be in (0, 1)")
        if not math.isfinite(self.voltage_tolerance_v) or self.voltage_tolerance_v < 0:
            raise OptimizationError("Voltage tolerance cannot be negative")
        if self.stability_samples <= 0:
            raise OptimizationError("Stability sample count must be positive")

    def settings(self) -> dict[str, object]:
        self.validate()
        settings = asdict(self)
        settings["thicknesses_um"] = list(self.thicknesses_um)
        settings["signed_target_current_density_a_m2"] = (
            self.signed_target_current_density_a_m2
        )
        return settings


def format_number(value: float) -> str:
    return f"{0.0 if abs(value) < 5.0e-14 else value:.12g}"


def case_label(thickness_um: float) -> str:
    number = format_number(thickness_um).replace("-", "m").replace(".", "p")
    return f"{number}um"


def _matching_brace(text: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise OptimizationError("Unmatched brace in OpenFOAM dictionary")


def _block_span(text: str, name: str, start: int = 0, end: int | None = None) -> tuple[int, int]:
    match = re.search(rf"\b{re.escape(name)}\b\s*\{{", text[start:end])
    if not match:
        raise OptimizationError(f"Cannot find {name!r} dictionary block")
    opening = start + match.start() + match.group(0).rfind("{")
    return opening + 1, _matching_brace(text, opening)


def _replace_scalar_in_span(
    text: str, start: int, end: int, keyword: str, value: str
) -> str:
    block = text[start:end]
    rewritten, count = re.subn(
        rf"(?m)^(\s*{re.escape(keyword)}\s+)[^;\n]+(\s*;)",
        lambda match: match.group(1) + value + match.group(2),
        block,
        count=1,
    )
    if count != 1:
        raise OptimizationError(f"Cannot update {keyword!r} in OpenFOAM dictionary")
    return text[:start] + rewritten + text[end:]


def _replace_table_values(
    text: str, block_start: int, block_end: int, pairs: tuple[tuple[float, float], ...]
) -> str:
    block = text[block_start:block_end]
    match = re.search(r"\bvalues\s*\(", block)
    if not match:
        raise OptimizationError("Cannot find table values in OpenFOAM dictionary")
    opening = block_start + match.start() + match.group(0).rfind("(")
    depth = 0
    closing = -1
    for index in range(opening, block_end):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                closing = index
                break
    if closing < 0:
        raise OptimizationError("Unmatched table values in OpenFOAM dictionary")
    indentation = "            "
    body = "\n" + "\n".join(
        f"{indentation}({format_number(time_s)}    {format_number(value)})"
        for time_s, value in pairs
    ) + "\n        "
    return text[: opening + 1] + body + text[closing:]


def _configure_electrical_control(path: Path, config: ExperimentConfig) -> None:
    text = path.read_text(encoding="utf-8")
    galvanostatic_start, galvanostatic_end = _block_span(text, "galvanostatic")
    text = _replace_scalar_in_span(
        text, galvanostatic_start, galvanostatic_end, "active", "true"
    )
    galvanostatic_start, galvanostatic_end = _block_span(text, "galvanostatic")
    ibar_start, ibar_end = _block_span(text, "ibar", galvanostatic_start, galvanostatic_end)
    target = config.signed_target_current_density_a_m2
    text = _replace_table_values(
        text, ibar_start, ibar_end, ((0.0, target), (config.duration_s, target))
    )
    galvanostatic_start, galvanostatic_end = _block_span(text, "galvanostatic")
    curve_start, curve_end = _block_span(
        text, "polarizationCurve", galvanostatic_start, galvanostatic_end
    )
    replacements = (
        ("active", "true"),
        ("targets", f"({format_number(target)})"),
        ("minimumHoldDuration", format_number(config.duration_s)),
        ("targetCurrentTolerance", format_number(config.target_current_tolerance)),
        ("voltageTolerance", format_number(config.voltage_tolerance_v)),
        ("currentStabilityTolerance", format_number(config.current_stability_tolerance)),
        ("stabilitySamples", str(config.stability_samples)),
    )
    for keyword, value in replacements:
        galvanostatic_start, galvanostatic_end = _block_span(text, "galvanostatic")
        curve_start, curve_end = _block_span(
            text, "polarizationCurve", galvanostatic_start, galvanostatic_end
        )
        text = _replace_scalar_in_span(text, curve_start, curve_end, keyword, value)
    path.write_text(text, encoding="utf-8")


def _replace_control_value(path: Path, keyword: str, value: float) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    rewritten, count = re.subn(
        rf"(?m)^(\s*{re.escape(keyword)}\s+)({FLOAT_PATTERN})(\s*;)",
        lambda match: match.group(1) + format_number(value) + match.group(3),
        text,
        count=1,
    )
    if count != 1:
        raise OptimizationError(f"Cannot update {keyword!r} in {path}")
    path.write_text(rewritten, encoding="utf-8")


def _replace_initial_field(path: Path, value: float) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    rewritten, count = re.subn(
        rf"(?m)^(\s*internalField\s+uniform\s+)({FLOAT_PATTERN})(\s*;)",
        lambda match: match.group(1) + format_number(value) + match.group(3),
        text,
        count=1,
    )
    if count != 1:
        raise OptimizationError(f"Cannot update initial potential in {path}")
    path.write_text(rewritten, encoding="utf-8")


def configure_case(case_path: Path, thickness_um: float, config: ExperimentConfig) -> None:
    """Configure one clean case for the prescribed thickness experiment."""
    config.validate()
    rewrite_block_mesh_thickness(case_path / "system/blockMeshDict", thickness_um)
    _configure_electrical_control(
        case_path / "constant/phiEAnode/regionProperties", config
    )
    for filename in ("controlDict.run", "controlDict"):
        path = case_path / "system" / filename
        _replace_control_value(path, "endTime", config.duration_s)
        _replace_control_value(path, "deltaT", config.delta_t_s)
        _replace_control_value(path, "writeInterval", config.duration_s)
    for directory in ("0.orig", "0"):
        _replace_initial_field(
            case_path / directory / "phiEAnode" / "phi", config.initial_voltage_v
        )

