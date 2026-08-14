"""Parse AEMEC logs into stable, reusable crossover data products."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable, Sequence

from .project import OptimizationError, parse_voltage_sweep_samples


FARADAY_CONSTANT_C_MOL = 96_485.33212
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TIME_RE = re.compile(rf"^\s*Time\s*=\s*({FLOAT_PATTERN})\s*$")
CONTROLLER_RE = re.compile(
    rf"\bgalvanostatic\s+target:\s*({FLOAT_PATTERN})\s*A/m2.*?"
    rf"current\s+relative\s+error:\s*({FLOAT_PATTERN}).*?"
    rf"stability\s+samples:\s*(\d+)\s*/\s*(\d+).*?"
    rf"accepted:\s*(true|false|1|0)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExperimentSample:
    time_s: float
    current_a: float
    current_density_a_m2: float
    voltage_v: float
    crossover_rate_mol_s: float
    target_current_density_a_m2: float | None = None
    current_relative_error: float | None = None
    stable_samples: int | None = None
    required_stable_samples: int | None = None
    accepted: bool | None = None

    @property
    def current_density_magnitude_a_cm2(self) -> float:
        return abs(self.current_density_a_m2) / 1.0e4

    @property
    def hydrogen_production_rate_mol_s(self) -> float:
        return abs(self.current_a) / (2.0 * FARADAY_CONSTANT_C_MOL)

    @property
    def crossover_fraction_percent(self) -> float:
        production = self.hydrogen_production_rate_mol_s
        return 100.0 * self.crossover_rate_mol_s / production if production > 0 else math.nan


@dataclass(frozen=True)
class ExperimentSummary:
    thickness_um: float
    final_time_s: float
    window_start_s: float
    window_sample_count: int
    mean_current_density_magnitude_a_cm2: float
    current_density_std_a_cm2: float
    mean_voltage_v: float
    voltage_std_v: float
    mean_crossover_rate_mol_s: float
    crossover_rate_std_mol_s: float
    mean_hydrogen_production_rate_mol_s: float
    mean_crossover_fraction_percent: float
    final_controller_accepted: bool | None

    def row(self) -> dict[str, object]:
        return asdict(self)


CSV_FIELDS = (
    "time_s",
    "current_a",
    "current_density_a_m2",
    "current_density_magnitude_a_cm2",
    "voltage_v",
    "crossover_rate_mol_s",
    "hydrogen_production_rate_mol_s",
    "crossover_fraction_percent",
    "target_current_density_a_m2",
    "current_relative_error",
    "stable_samples",
    "required_stable_samples",
    "accepted",
)


def _controller_metadata(log_text: str) -> dict[float, tuple[float, float, int, int, bool]]:
    current_time: float | None = None
    metadata: dict[float, tuple[float, float, int, int, bool]] = {}
    for line in log_text.splitlines():
        time_match = TIME_RE.search(line)
        if time_match:
            current_time = float(time_match.group(1))
            continue
        match = CONTROLLER_RE.search(line)
        if match and current_time is not None:
            values = (
                float(match.group(1)),
                float(match.group(2)),
                int(match.group(3)),
                int(match.group(4)),
                match.group(5).lower() in {"true", "1"},
            )
            metadata[round(current_time, 12)] = values
    return metadata


def parse_experiment_log(log_text: str) -> list[ExperimentSample]:
    """Pair current, voltage, crossover, and controller state by time step."""
    metadata = _controller_metadata(log_text)
    samples: list[ExperimentSample] = []
    for raw in parse_voltage_sweep_samples(log_text):
        control = metadata.get(round(raw.time_s, 12))
        if control is None:
            target = relative_error = stable = required = accepted = None
        else:
            target, relative_error, stable, required, accepted = control
        sample = ExperimentSample(
            time_s=raw.time_s,
            current_a=raw.current_a,
            current_density_a_m2=raw.current_density_a_m2,
            voltage_v=raw.cell_voltage_v,
            crossover_rate_mol_s=raw.crossover_rate_mol_s,
            target_current_density_a_m2=target,
            current_relative_error=relative_error,
            stable_samples=stable,
            required_stable_samples=required,
            accepted=accepted,
        )
        if sample.crossover_rate_mol_s < 0:
            raise OptimizationError(
                f"Negative crossover rate at t={sample.time_s:g} s"
            )
        samples.append(sample)
    if not samples:
        raise OptimizationError(
            "No paired current/voltage and hydrogen-crossover samples were found"
        )
    return samples


def _optional_number(value: str, converter: type[float] | type[int]) -> float | int | None:
    return None if value.strip() == "" else converter(value)


def write_timeseries_csv(path: Path, samples: Sequence[ExperimentSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "time_s": sample.time_s,
                    "current_a": sample.current_a,
                    "current_density_a_m2": sample.current_density_a_m2,
                    "current_density_magnitude_a_cm2": sample.current_density_magnitude_a_cm2,
                    "voltage_v": sample.voltage_v,
                    "crossover_rate_mol_s": sample.crossover_rate_mol_s,
                    "hydrogen_production_rate_mol_s": sample.hydrogen_production_rate_mol_s,
                    "crossover_fraction_percent": sample.crossover_fraction_percent,
                    "target_current_density_a_m2": "" if sample.target_current_density_a_m2 is None else sample.target_current_density_a_m2,
                    "current_relative_error": "" if sample.current_relative_error is None else sample.current_relative_error,
                    "stable_samples": "" if sample.stable_samples is None else sample.stable_samples,
                    "required_stable_samples": "" if sample.required_stable_samples is None else sample.required_stable_samples,
                    "accepted": "" if sample.accepted is None else str(sample.accepted).lower(),
                }
            )


def read_timeseries_csv(path: Path) -> list[ExperimentSample]:
    samples: list[ExperimentSample] = []
    try:
        with path.open(newline="", encoding="utf-8") as input_file:
            for row in csv.DictReader(input_file):
                accepted_text = row.get("accepted", "").strip().lower()
                samples.append(
                    ExperimentSample(
                        time_s=float(row["time_s"]),
                        current_a=float(row["current_a"]),
                        current_density_a_m2=float(row["current_density_a_m2"]),
                        voltage_v=float(row["voltage_v"]),
                        crossover_rate_mol_s=float(row["crossover_rate_mol_s"]),
                        target_current_density_a_m2=_optional_number(row.get("target_current_density_a_m2", ""), float),
                        current_relative_error=_optional_number(row.get("current_relative_error", ""), float),
                        stable_samples=_optional_number(row.get("stable_samples", ""), int),
                        required_stable_samples=_optional_number(row.get("required_stable_samples", ""), int),
                        accepted=None if not accepted_text else accepted_text in {"true", "1", "yes"},
                    )
                )
    except (KeyError, TypeError, ValueError) as exc:
        raise OptimizationError(f"Invalid experiment time-series CSV: {path}") from exc
    if not samples:
        raise OptimizationError(f"Experiment time-series CSV is empty: {path}")
    return samples


def summarize_samples(
    thickness_um: float,
    samples: Sequence[ExperimentSample],
    final_window_s: float,
) -> ExperimentSummary:
    if not samples:
        raise OptimizationError("Cannot summarize an empty experiment")
    ordered = sorted(samples, key=lambda sample: sample.time_s)
    final_time = ordered[-1].time_s
    window_start = max(ordered[0].time_s, final_time - final_window_s)
    window = [sample for sample in ordered if sample.time_s >= window_start - 1.0e-12]
    if not window:
        raise OptimizationError("The final averaging window contains no samples")

    currents = [sample.current_density_magnitude_a_cm2 for sample in window]
    voltages = [sample.voltage_v for sample in window]
    crossovers = [sample.crossover_rate_mol_s for sample in window]
    productions = [sample.hydrogen_production_rate_mol_s for sample in window]
    fractions = [sample.crossover_fraction_percent for sample in window]
    finite_fractions = [value for value in fractions if math.isfinite(value)]
    final_accepted = next(
        (sample.accepted for sample in reversed(ordered) if sample.accepted is not None),
        None,
    )
    return ExperimentSummary(
        thickness_um=thickness_um,
        final_time_s=final_time,
        window_start_s=window_start,
        window_sample_count=len(window),
        mean_current_density_magnitude_a_cm2=fmean(currents),
        current_density_std_a_cm2=pstdev(currents),
        mean_voltage_v=fmean(voltages),
        voltage_std_v=pstdev(voltages),
        mean_crossover_rate_mol_s=fmean(crossovers),
        crossover_rate_std_mol_s=pstdev(crossovers),
        mean_hydrogen_production_rate_mol_s=fmean(productions),
        mean_crossover_fraction_percent=(
            fmean(finite_fractions) if finite_fractions else math.nan
        ),
        final_controller_accepted=final_accepted,
    )


def validate_completed_run(
    samples: Sequence[ExperimentSample], duration_s: float, delta_t_s: float
) -> None:
    final_time = max(sample.time_s for sample in samples)
    tolerance = max(1.0e-8, 0.51 * delta_t_s)
    if final_time < duration_s - tolerance:
        raise OptimizationError(
            f"Final paired sample is at {final_time:g} s; expected {duration_s:g} s"
        )

