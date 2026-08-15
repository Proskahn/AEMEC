"""Scientific records for the current-density crossover sweep."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import fmean, pstdev
from typing import Sequence

from .config import CurrentSweepConfig
from .parsing import FARADAY_CONSTANT_C_MOL, ExperimentSample
from .project import OptimizationError


@dataclass(frozen=True)
class CurrentSweepPoint:
    target_current_density_a_cm2: float
    signed_target_current_density_a_m2: float
    accepted_time_s: float
    window_start_s: float
    window_sample_count: int
    mean_measured_current_density_a_cm2: float
    current_density_std_a_cm2: float
    mean_voltage_v: float
    voltage_std_v: float
    mean_crossover_rate_mol_s: float
    crossover_rate_std_mol_s: float
    membrane_area_m2: float
    mean_crossover_flux_density_mol_m2_s: float
    crossover_flux_density_std_mol_m2_s: float
    hydrogen_production_flux_density_mol_m2_s: float
    crossover_fraction_percent: float | None

    def row(self) -> dict[str, object]:
        values = asdict(self)
        if self.crossover_fraction_percent is None:
            values["crossover_fraction_percent"] = ""
        return values


TIMESERIES_FIELDS = (
    "time_s",
    "target_current_density_a_cm2",
    "signed_target_current_density_a_m2",
    "measured_current_density_a_cm2",
    "signed_measured_current_density_a_m2",
    "current_a",
    "voltage_v",
    "crossover_rate_mol_s",
    "membrane_area_m2",
    "crossover_flux_density_mol_m2_s",
    "stable_samples",
    "required_stable_samples",
    "accepted",
)


def target_a_cm2(sample: ExperimentSample) -> float | None:
    if sample.target_current_density_a_m2 is None:
        return None
    return abs(sample.target_current_density_a_m2) / 1.0e4


def crossover_flux_density(sample: ExperimentSample, membrane_area_m2: float) -> float:
    return sample.crossover_rate_mol_s / membrane_area_m2


def write_current_sweep_timeseries(
    path: Path,
    samples: Sequence[ExperimentSample],
    membrane_area_m2: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=TIMESERIES_FIELDS)
        writer.writeheader()
        for sample in samples:
            target = target_a_cm2(sample)
            writer.writerow(
                {
                    "time_s": sample.time_s,
                    "target_current_density_a_cm2": "" if target is None else target,
                    "signed_target_current_density_a_m2": "" if sample.target_current_density_a_m2 is None else sample.target_current_density_a_m2,
                    "measured_current_density_a_cm2": sample.current_density_magnitude_a_cm2,
                    "signed_measured_current_density_a_m2": sample.current_density_a_m2,
                    "current_a": sample.current_a,
                    "voltage_v": sample.voltage_v,
                    "crossover_rate_mol_s": sample.crossover_rate_mol_s,
                    "membrane_area_m2": membrane_area_m2,
                    "crossover_flux_density_mol_m2_s": crossover_flux_density(sample, membrane_area_m2),
                    "stable_samples": "" if sample.stable_samples is None else sample.stable_samples,
                    "required_stable_samples": "" if sample.required_stable_samples is None else sample.required_stable_samples,
                    "accepted": "" if sample.accepted is None else str(sample.accepted).lower(),
                }
            )


def _optional_int(value: str) -> int | None:
    return None if not value.strip() else int(value)


def read_current_sweep_timeseries(path: Path) -> list[ExperimentSample]:
    samples: list[ExperimentSample] = []
    try:
        with path.open(newline="", encoding="utf-8") as input_file:
            for row in csv.DictReader(input_file):
                accepted_text = row.get("accepted", "").strip().lower()
                target_text = row.get("signed_target_current_density_a_m2", "").strip()
                samples.append(
                    ExperimentSample(
                        time_s=float(row["time_s"]),
                        current_a=float(row["current_a"]),
                        current_density_a_m2=float(row["signed_measured_current_density_a_m2"]),
                        voltage_v=float(row["voltage_v"]),
                        crossover_rate_mol_s=float(row["crossover_rate_mol_s"]),
                        target_current_density_a_m2=(None if not target_text else float(target_text)),
                        stable_samples=_optional_int(row.get("stable_samples", "")),
                        required_stable_samples=_optional_int(row.get("required_stable_samples", "")),
                        accepted=(None if not accepted_text else accepted_text in {"true", "1", "yes"}),
                    )
                )
    except (KeyError, TypeError, ValueError) as exc:
        raise OptimizationError(f"Invalid current-sweep time-series CSV: {path}") from exc
    if not samples:
        raise OptimizationError(f"Current-sweep time-series CSV is empty: {path}")
    return samples


def _verify_membrane_area(samples: Sequence[ExperimentSample], expected_area_m2: float) -> None:
    inferred = [
        abs(sample.current_a / sample.current_density_a_m2)
        for sample in samples
        if abs(sample.current_density_a_m2) > 1.0e-12
    ]
    if not inferred:
        return
    inferred_area = fmean(inferred)
    if abs(inferred_area - expected_area_m2) / expected_area_m2 > 0.01:
        raise OptimizationError(
            "Controlled-boundary area does not match the configured membrane area: "
            f"log={inferred_area:.8g} m2, configured={expected_area_m2:.8g} m2"
        )


def summarize_current_sweep(
    samples: Sequence[ExperimentSample], config: CurrentSweepConfig
) -> list[CurrentSweepPoint]:
    config.validate()
    if not samples:
        raise OptimizationError("Cannot summarize an empty current sweep")
    _verify_membrane_area(samples, config.membrane_area_m2)
    grouped: dict[float, list[ExperimentSample]] = {}
    for sample in samples:
        target = target_a_cm2(sample)
        if target is not None:
            grouped.setdefault(round(target, 9), []).append(sample)

    points: list[CurrentSweepPoint] = []
    for target, signed_target in zip(
        config.current_targets_a_cm2, config.signed_targets_a_m2
    ):
        target_samples = sorted(grouped.get(round(target, 9), []), key=lambda item: item.time_s)
        accepted = [sample for sample in target_samples if sample.accepted is True]
        if not accepted:
            raise OptimizationError(
                f"Current target {target:g} A/cm2 was not accepted before the sweep ended"
            )
        endpoint = accepted[-1]
        required_samples = (
            endpoint.required_stable_samples or config.stability_samples
        )
        eligible = [
            sample
            for sample in target_samples
            if sample.time_s <= endpoint.time_s + 1.0e-12
        ]
        if len(eligible) < required_samples:
            raise OptimizationError(
                f"Current target {target:g} A/cm2 has only {len(eligible)} samples "
                f"before acceptance; {required_samples} are required"
            )
        if endpoint.stable_samples is not None and endpoint.stable_samples < required_samples:
            raise OptimizationError(
                f"Current target {target:g} A/cm2 was marked accepted with only "
                f"{endpoint.stable_samples}/{required_samples} stable samples"
            )
        # The controller accepts a target after this exact number of consecutive
        # stable samples.  Averaging an arbitrary time interval before acceptance
        # would mix in the preceding voltage/current convergence transient.
        window = eligible[-required_samples:]
        window_start = window[0].time_s
        measured = [sample.current_density_magnitude_a_cm2 for sample in window]
        voltage = [sample.voltage_v for sample in window]
        rates = [sample.crossover_rate_mol_s for sample in window]
        fluxes = [rate / config.membrane_area_m2 for rate in rates]
        mean_measured = fmean(measured)
        target_scale_a_cm2 = max(target, config.zero_current_scale_a_m2 / 1.0e4)
        relative_error = abs(mean_measured - target) / target_scale_a_cm2
        if relative_error > config.target_current_tolerance:
            raise OptimizationError(
                f"Accepted stability-window current at {target:g} A/cm2 has relative error "
                f"{relative_error:.6g}; limit={config.target_current_tolerance:.6g}"
            )
        production_flux = target * 1.0e4 / (2.0 * FARADAY_CONSTANT_C_MOL)
        mean_flux = fmean(fluxes)
        fraction = None if production_flux <= 0 else 100.0 * mean_flux / production_flux
        points.append(
            CurrentSweepPoint(
                target_current_density_a_cm2=target,
                signed_target_current_density_a_m2=signed_target,
                accepted_time_s=endpoint.time_s,
                window_start_s=window_start,
                window_sample_count=len(window),
                mean_measured_current_density_a_cm2=mean_measured,
                current_density_std_a_cm2=pstdev(measured),
                mean_voltage_v=fmean(voltage),
                voltage_std_v=pstdev(voltage),
                mean_crossover_rate_mol_s=fmean(rates),
                crossover_rate_std_mol_s=pstdev(rates),
                membrane_area_m2=config.membrane_area_m2,
                mean_crossover_flux_density_mol_m2_s=mean_flux,
                crossover_flux_density_std_mol_m2_s=pstdev(fluxes),
                hydrogen_production_flux_density_mol_m2_s=production_flux,
                crossover_fraction_percent=fraction,
            )
        )
    return points


def write_current_sweep_points(path: Path, points: Sequence[CurrentSweepPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(CurrentSweepPoint)]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(point.row())
