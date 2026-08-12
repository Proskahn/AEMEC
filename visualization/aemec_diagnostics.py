#!/usr/bin/env python3
"""Plot AEMEC voltage losses and hydrogen crossover from an OpenFOAM log.

The voltage decomposition uses current-weighted diagnostics emitted by the
reaction and electric regions.  Hydrogen crossover is shown both as a molar
rate and as a fraction of the Faradaic hydrogen-production rate.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from visualization.polarization_curve import (
        DEFAULT_ACTIVE_AREA_CM2,
        Sample,
        last_sample_per_hold,
        last_sample_per_target_current,
        last_sample_per_voltage,
        log_ended_normally,
        parse_log,
    )
except ModuleNotFoundError:  # Direct execution from the visualization folder.
    from polarization_curve import (  # type: ignore[no-redef]
        DEFAULT_ACTIVE_AREA_CM2,
        Sample,
        last_sample_per_hold,
        last_sample_per_target_current,
        last_sample_per_voltage,
        log_ended_normally,
        parse_log,
    )


FLOAT_PATTERN = r"[-+0-9.eE]+"
TIME_RE = re.compile(rf"^Time\s*=\s*({FLOAT_PATTERN})")
ZONE_RE = re.compile(r"\bzone=([^,\s]+)")
REGION_RE = re.compile(r"\bregion=([^,\s]+)")
REACTION_CURRENT_RE = re.compile(rf"\breactionCurrent=({FLOAT_PATTERN})\s*A")
ACTIVATION_VOLTAGE_RE = re.compile(
    rf"\bequivalentActivationVoltage=({FLOAT_PATTERN})\s*V"
)
CURRENT_WEIGHTED_NERNST_RE = re.compile(
    rf"\bcurrentWeightedNernst=({FLOAT_PATTERN})\s*V"
)
ETA_RE = re.compile(
    rf"\beta\[min,mean,max\]=\(({FLOAT_PATTERN}),({FLOAT_PATTERN}),"
    rf"({FLOAT_PATTERN})\)\s*V"
)
NERNST_RE = re.compile(
    rf"\bnernst\[min,mean,max\]=\(({FLOAT_PATTERN}),({FLOAT_PATTERN}),"
    rf"({FLOAT_PATTERN})\)\s*V"
)
OHMIC_POWER_RE = re.compile(rf"\bohmicPower=({FLOAT_PATTERN})\s*W")
OHMIC_VOLTAGE_RE = re.compile(
    rf"\bequivalentOhmicVoltage=({FLOAT_PATTERN})\s*V"
)
CROSSOVER_OBJECTIVE_RE = re.compile(
    rf"Hydrogen crossover objective: anode gas source rate = "
    rf"({FLOAT_PATTERN})\s*mol/s"
)
OLD_FARADAIC_RE = re.compile(
    rf"Hydrogen crossover membrane diagnostic:.*?"
    rf"derived Faradaic H2 source = ({FLOAT_PATTERN})\s*mol/s"
)
COUPLED_PARTITION_RE = re.compile(
    rf"Hydrogen production partition:.*?"
    rf"Faradaic cathode H2 generation = ({FLOAT_PATTERN})\s*mol/s.*?"
    rf"initially dissolved = ({FLOAT_PATTERN})\s*mol/s.*?"
    rf"direct Faradaic gas = ({FLOAT_PATTERN})\s*mol/s.*?"
    rf"cathode dissolved-to-gas transfer = ({FLOAT_PATTERN})\s*mol/s.*?"
    rf"anode dissolved-to-gas transfer = ({FLOAT_PATTERN})\s*mol/s.*?"
    rf"dissolved inventory = ({FLOAT_PATTERN})\s*mol"
)
LEGACY_PARTITION_RE = re.compile(
    rf"Hydrogen production partition:.*?"
    rf"Faradaic cathode H2 generation = ({FLOAT_PATTERN})\s*mol/s.*?"
    rf"membrane crossover = ({FLOAT_PATTERN})\s*mol/s.*?"
    rf"retained in cathode gas = ({FLOAT_PATTERN})\s*mol/s"
)

FARADAY_C_PER_MOL = 96485.33212


@dataclass(frozen=True)
class ReactionDiagnostic:
    zone: str
    current_a: float
    activation_voltage_v: float
    nernst_voltage_v: float


@dataclass(frozen=True)
class ElectricDiagnostic:
    region: str
    ohmic_power_w: float
    ohmic_voltage_v: float


@dataclass
class TimeDiagnostics:
    reactions: dict[str, ReactionDiagnostic] = field(default_factory=dict)
    electric: dict[str, ElectricDiagnostic] = field(default_factory=dict)
    crossover_rate_mol_s: float | None = None
    faradaic_h2_rate_mol_s: float | None = None
    dissolved_production_rate_mol_s: float | None = None
    direct_faradaic_gas_rate_mol_s: float | None = None
    cathode_transfer_rate_mol_s: float | None = None
    anode_transfer_rate_mol_s: float | None = None
    cathode_gas_release_rate_mol_s: float | None = None
    dissolved_inventory_mol: float | None = None


@dataclass(frozen=True)
class VoltagePoint:
    time_s: float
    current_density_a_cm2: float
    cell_voltage_v: float
    reversible_voltage_v: float
    anode_activation_v: float
    cathode_activation_v: float
    electronic_ohmic_v: float
    anion_ohmic_v: float
    unresolved_v: float


@dataclass(frozen=True)
class CrossoverPoint:
    time_s: float
    current_density_a_cm2: float
    cell_voltage_v: float
    faradaic_h2_rate_mol_s: float
    dissolved_production_rate_mol_s: float
    direct_faradaic_gas_rate_mol_s: float
    cathode_transfer_rate_mol_s: float
    cathode_gas_release_rate_mol_s: float
    anode_transfer_rate_mol_s: float
    dissolved_inventory_mol: float
    crossover_rate_mol_s: float
    crossover_fraction_percent: float


def _required_match(pattern: re.Pattern[str], line: str) -> re.Match[str] | None:
    return pattern.search(line)


def parse_diagnostics(log_path: Path) -> dict[float, TimeDiagnostics]:
    """Extract the last diagnostic record written at each solver time."""
    diagnostics: dict[float, TimeDiagnostics] = {}
    current_time: float | None = None

    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            time_match = TIME_RE.search(line)
            if time_match:
                current_time = float(time_match.group(1))
                diagnostics.setdefault(current_time, TimeDiagnostics())
                continue
            if current_time is None:
                continue

            current = diagnostics[current_time]

            if "AEMEC reaction diagnostic:" in line:
                zone_match = _required_match(ZONE_RE, line)
                current_match = _required_match(REACTION_CURRENT_RE, line)
                eta_match = _required_match(ETA_RE, line)
                nernst_match = _required_match(NERNST_RE, line)
                if not all((zone_match, current_match, eta_match, nernst_match)):
                    continue
                assert zone_match and current_match and eta_match and nernst_match
                activation_match = ACTIVATION_VOLTAGE_RE.search(line)
                weighted_nernst_match = CURRENT_WEIGHTED_NERNST_RE.search(line)
                activation_voltage = (
                    float(activation_match.group(1))
                    if activation_match
                    else abs(float(eta_match.group(2)))
                )
                nernst_voltage = (
                    float(weighted_nernst_match.group(1))
                    if weighted_nernst_match
                    else float(nernst_match.group(2))
                )
                zone = zone_match.group(1)
                current.reactions[zone] = ReactionDiagnostic(
                    zone=zone,
                    current_a=float(current_match.group(1)),
                    activation_voltage_v=activation_voltage,
                    nernst_voltage_v=nernst_voltage,
                )
                continue

            if "AEMEC electric diagnostic:" in line:
                region_match = _required_match(REGION_RE, line)
                power_match = _required_match(OHMIC_POWER_RE, line)
                voltage_match = _required_match(OHMIC_VOLTAGE_RE, line)
                if not all((region_match, power_match, voltage_match)):
                    continue
                assert region_match and power_match and voltage_match
                region = region_match.group(1)
                current.electric[region] = ElectricDiagnostic(
                    region=region,
                    ohmic_power_w=float(power_match.group(1)),
                    ohmic_voltage_v=float(voltage_match.group(1)),
                )
                continue

            partition_match = COUPLED_PARTITION_RE.search(line)
            if partition_match:
                current.faradaic_h2_rate_mol_s = float(partition_match.group(1))
                current.dissolved_production_rate_mol_s = float(
                    partition_match.group(2)
                )
                current.direct_faradaic_gas_rate_mol_s = float(
                    partition_match.group(3)
                )
                current.cathode_transfer_rate_mol_s = float(
                    partition_match.group(4)
                )
                current.anode_transfer_rate_mol_s = float(
                    partition_match.group(5)
                )
                current.dissolved_inventory_mol = float(partition_match.group(6))
                current.cathode_gas_release_rate_mol_s = (
                    current.direct_faradaic_gas_rate_mol_s
                    + current.cathode_transfer_rate_mol_s
                )
                current.crossover_rate_mol_s = max(
                    current.anode_transfer_rate_mol_s, 0.0
                )
                continue

            partition_match = LEGACY_PARTITION_RE.search(line)
            if partition_match:
                current.faradaic_h2_rate_mol_s = float(partition_match.group(1))
                current.crossover_rate_mol_s = float(partition_match.group(2))
                current.anode_transfer_rate_mol_s = current.crossover_rate_mol_s
                current.cathode_gas_release_rate_mol_s = float(
                    partition_match.group(3)
                )
                continue

            crossover_match = CROSSOVER_OBJECTIVE_RE.search(line)
            if crossover_match:
                current.crossover_rate_mol_s = float(crossover_match.group(1))
                continue

            faradaic_match = OLD_FARADAIC_RE.search(line)
            if faradaic_match:
                current.faradaic_h2_rate_mol_s = float(faradaic_match.group(1))

    return diagnostics


def select_curve_samples(
    samples: list[Sample],
    scan_mode: str,
    hold_duration: float,
    voltage_precision: int,
    current_precision: int,
    all_samples: bool,
) -> list[Sample]:
    controller_samples = [sample for sample in samples if sample.accepted is not None]
    if controller_samples and not all_samples:
        samples = [sample for sample in controller_samples if sample.accepted]
        if not samples:
            raise ValueError("no accepted stable polarization points were logged")

    if all_samples:
        return samples

    resolved_mode = scan_mode
    if resolved_mode == "auto":
        resolved_mode = "current" if controller_samples else "voltage"

    if resolved_mode == "voltage":
        return last_sample_per_voltage(samples, voltage_precision)

    try:
        return last_sample_per_target_current(samples, current_precision)
    except ValueError:
        return last_sample_per_hold(samples, hold_duration)


def build_voltage_points(
    samples: list[Sample], diagnostics: dict[float, TimeDiagnostics]
) -> tuple[list[VoltagePoint], list[float]]:
    points: list[VoltagePoint] = []
    missing_times: list[float] = []

    for sample in samples:
        if sample.time is None or sample.time not in diagnostics:
            if sample.time is not None:
                missing_times.append(sample.time)
            continue
        diagnostic = diagnostics[sample.time]
        required_reactions = {"anodeCL", "cathodeCL"}
        required_electric = {"phiEAnode", "phiECathode", "phiAnion"}
        if not required_reactions.issubset(diagnostic.reactions) or not required_electric.issubset(
            diagnostic.electric
        ):
            missing_times.append(sample.time)
            continue

        anode = diagnostic.reactions["anodeCL"]
        cathode = diagnostic.reactions["cathodeCL"]
        reversible = abs(anode.nernst_voltage_v - cathode.nernst_voltage_v)
        electronic_ohmic = (
            diagnostic.electric["phiEAnode"].ohmic_voltage_v
            + diagnostic.electric["phiECathode"].ohmic_voltage_v
        )
        anion_ohmic = diagnostic.electric["phiAnion"].ohmic_voltage_v
        accounted = (
            reversible
            + anode.activation_voltage_v
            + cathode.activation_voltage_v
            + electronic_ohmic
            + anion_ohmic
        )
        points.append(
            VoltagePoint(
                time_s=sample.time,
                current_density_a_cm2=abs(sample.current_density_a_cm2),
                cell_voltage_v=sample.voltage_v,
                reversible_voltage_v=reversible,
                anode_activation_v=anode.activation_voltage_v,
                cathode_activation_v=cathode.activation_voltage_v,
                electronic_ohmic_v=electronic_ohmic,
                anion_ohmic_v=anion_ohmic,
                unresolved_v=sample.voltage_v - accounted,
            )
        )

    return sorted(points, key=lambda point: point.current_density_a_cm2), missing_times


def build_crossover_points(
    samples: list[Sample],
    diagnostics: dict[float, TimeDiagnostics],
) -> tuple[list[CrossoverPoint], list[float]]:
    points: list[CrossoverPoint] = []
    missing_times: list[float] = []

    for sample in samples:
        if sample.time is None or sample.time not in diagnostics:
            if sample.time is not None:
                missing_times.append(sample.time)
            continue
        diagnostic = diagnostics[sample.time]
        crossover = diagnostic.crossover_rate_mol_s
        if crossover is None:
            missing_times.append(sample.time)
            continue

        faradaic = diagnostic.faradaic_h2_rate_mol_s
        if faradaic is None:
            current_a = (
                abs(sample.current_a)
                if sample.current_a is not None
                else 0.0
            )
            faradaic = current_a/(2.0*FARADAY_C_PER_MOL)
        cathode_gas_release = diagnostic.cathode_gas_release_rate_mol_s
        if cathode_gas_release is None:
            # Compatibility fallback for logs written before the dissolved-H2
            # inventory and signed CL transfer rates were reported.
            cathode_gas_release = faradaic - crossover
        # At (near) open circuit, crossover can be supplied by the existing
        # dissolved-H2 inventory rather than by instantaneous Faradaic
        # production. A "fraction of production" is not meaningful when the
        # crossover rate exceeds that production rate, so leave it undefined.
        fraction = (
            100.0*crossover/faradaic
            if faradaic > 1.0e-20 and 0.0 <= crossover <= faradaic
            else math.nan
        )

        points.append(
            CrossoverPoint(
                time_s=sample.time,
                current_density_a_cm2=abs(sample.current_density_a_cm2),
                cell_voltage_v=sample.voltage_v,
                faradaic_h2_rate_mol_s=faradaic,
                dissolved_production_rate_mol_s=(
                    diagnostic.dissolved_production_rate_mol_s
                    if diagnostic.dissolved_production_rate_mol_s is not None
                    else math.nan
                ),
                direct_faradaic_gas_rate_mol_s=(
                    diagnostic.direct_faradaic_gas_rate_mol_s
                    if diagnostic.direct_faradaic_gas_rate_mol_s is not None
                    else math.nan
                ),
                cathode_transfer_rate_mol_s=(
                    diagnostic.cathode_transfer_rate_mol_s
                    if diagnostic.cathode_transfer_rate_mol_s is not None
                    else math.nan
                ),
                cathode_gas_release_rate_mol_s=cathode_gas_release,
                anode_transfer_rate_mol_s=(
                    diagnostic.anode_transfer_rate_mol_s
                    if diagnostic.anode_transfer_rate_mol_s is not None
                    else crossover
                ),
                dissolved_inventory_mol=(
                    diagnostic.dissolved_inventory_mol
                    if diagnostic.dissolved_inventory_mol is not None
                    else math.nan
                ),
                crossover_rate_mol_s=crossover,
                crossover_fraction_percent=fraction,
            )
        )

    return sorted(points, key=lambda point: point.current_density_a_cm2), missing_times


def write_voltage_csv(points: list[VoltagePoint], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(VoltagePoint.__dataclass_fields__)
        for point in points:
            writer.writerow(point.__dict__.values())


def write_crossover_csv(points: list[CrossoverPoint], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(CrossoverPoint.__dataclass_fields__)
        for point in points:
            writer.writerow(point.__dict__.values())


def plot_voltage_decomposition(points: list[VoltagePoint], path: Path) -> None:
    if not points:
        raise ValueError("no complete voltage-decomposition points are available")

    current = [point.current_density_a_cm2 for point in points]
    reversible_reference = points[0].reversible_voltage_v
    series = (
        ("Cell voltage", [point.cell_voltage_v for point in points], "o", 2.4),
        (
            "Reversible baseline (lowest current)",
            [reversible_reference]*len(points),
            None,
            1.8,
        ),
        (
            "Nernst shift (transport proxy)",
            [point.reversible_voltage_v - reversible_reference for point in points],
            "s",
            1.6,
        ),
        ("Anode activation", [point.anode_activation_v for point in points], "^", 1.6),
        ("Cathode activation", [point.cathode_activation_v for point in points], "v", 1.6),
        ("Electronic ohmic", [point.electronic_ohmic_v for point in points], "D", 1.6),
        ("Anion ohmic", [point.anion_ohmic_v for point in points], "P", 1.6),
        ("Unresolved closure", [point.unresolved_v for point in points], "x", 1.4),
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    for label, values, marker, width in series:
        ax.plot(current, values, label=label, marker=marker, linewidth=width)
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_xlabel("|Current density| [A/cm²]")
    ax.set_ylabel("Voltage contribution [V]")
    ax.set_title("AEMEC voltage decomposition")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    ax.ticklabel_format(axis="x", style="plain")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_hydrogen_crossover(points: list[CrossoverPoint], path: Path) -> None:
    if not points:
        raise ValueError("no hydrogen-crossover points are available")

    current = [point.current_density_a_cm2 for point in points]
    fig, (rate_ax, fraction_ax) = plt.subplots(
        2,
        1,
        figsize=(7.6, 7.0),
        sharex=True,
        constrained_layout=True,
    )
    rate_series = (
        (
            "Faradaic H₂ generation",
            [point.faradaic_h2_rate_mol_s for point in points],
            "o",
            1.9,
        ),
        (
            "Initially dissolved",
            [point.dissolved_production_rate_mol_s for point in points],
            "D",
            1.3,
        ),
        (
            "Cathode dissolved→gas transfer",
            [point.cathode_transfer_rate_mol_s for point in points],
            "v",
            1.3,
        ),
        (
            "Net cathode gas release",
            [point.cathode_gas_release_rate_mol_s for point in points],
            "s",
            1.7,
        ),
        (
            "Anode gas release (crossover)",
            [point.crossover_rate_mol_s for point in points],
            "^",
            1.7,
        ),
    )
    for label, values, marker, linewidth in rate_series:
        if any(math.isfinite(rate) for rate in values):
            rate_ax.plot(
                current,
                values,
                marker=marker,
                linewidth=linewidth,
                label=label,
            )
    rate_ax.set_ylabel("H₂ rate [mol/s]")
    rate_ax.set_title("AEMEC hydrogen production and crossover")
    positive_rates = [
        rate
        for _, values, _, _ in rate_series
        for rate in values
        if rate > 0.0 and math.isfinite(rate)
    ]
    finite_rates = [
        rate
        for _, values, _, _ in rate_series
        for rate in values
        if math.isfinite(rate)
    ]
    if (
        positive_rates
        and len(positive_rates) == len(finite_rates)
        and max(positive_rates)/min(positive_rates) > 100.0
    ):
        rate_ax.set_yscale("log")
    else:
        rate_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    rate_ax.grid(True, alpha=0.25)
    rate_ax.legend(fontsize=8)

    fraction_ax.plot(
        current,
        [point.crossover_fraction_percent for point in points],
        marker="o",
        linewidth=1.9,
    )
    fraction_ax.set_xlabel("|Current density| [A/cm²]")
    fraction_ax.set_ylabel("Crossover fraction [%]")
    fraction_ax.grid(True, alpha=0.25)
    fraction_ax.ticklabel_format(axis="x", style="plain")
    if any(math.isnan(point.crossover_fraction_percent) for point in points):
        fraction_ax.text(
            0.01,
            0.96,
            "Fraction omitted where crossover exceeds instantaneous production",
            transform=fraction_ax.transAxes,
            va="top",
            fontsize=8,
        )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("run/AEMEC/log.run"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("visualization/output")
    )
    parser.add_argument(
        "--scan-mode", choices=("auto", "current", "voltage"), default="auto"
    )
    parser.add_argument("--hold-duration", type=float, default=15.0)
    parser.add_argument("--voltage-precision", type=int, default=3)
    parser.add_argument("--current-precision", type=int, default=3)
    parser.add_argument(
        "--active-area-cm2", type=float, default=DEFAULT_ACTIVE_AREA_CM2
    )
    parser.add_argument("--all-samples", action="store_true")
    parser.add_argument("--allow-incomplete-log", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    log_path = args.log.resolve()
    output_dir = args.output_dir.resolve()

    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")
    if not args.allow_incomplete_log and not log_ended_normally(log_path):
        raise SystemExit(
            "Solver log has no normal OpenFOAM 'End'; use "
            "--allow-incomplete-log only to inspect transients."
        )
    if args.hold_duration <= 0.0 or args.active_area_cm2 <= 0.0:
        raise SystemExit("hold duration and active area must be positive")

    samples = parse_log(log_path, args.active_area_cm2)
    if not samples:
        raise SystemExit(f"No polarization samples found in: {log_path}")
    try:
        selected = select_curve_samples(
            samples,
            args.scan_mode,
            args.hold_duration,
            args.voltage_precision,
            args.current_precision,
            args.all_samples,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    diagnostics = parse_diagnostics(log_path)
    voltage_points, missing_voltage = build_voltage_points(selected, diagnostics)
    crossover_points, missing_crossover = build_crossover_points(selected, diagnostics)
    output_dir.mkdir(parents=True, exist_ok=True)

    if voltage_points:
        voltage_csv = output_dir/"voltage_decomposition.csv"
        voltage_plot = output_dir/"voltage_decomposition.png"
        write_voltage_csv(voltage_points, voltage_csv)
        plot_voltage_decomposition(voltage_points, voltage_plot)
        print(f"Wrote voltage plot: {voltage_plot}")
        print(f"Wrote voltage data: {voltage_csv}")
    else:
        print(
            "Skipped voltage decomposition: the log has no complete reaction "
            "and electric diagnostics. Rebuild and rerun AEMEC with diagnostics enabled."
        )

    if crossover_points:
        crossover_csv = output_dir/"hydrogen_crossover.csv"
        crossover_plot = output_dir/"hydrogen_crossover.png"
        write_crossover_csv(crossover_points, crossover_csv)
        plot_hydrogen_crossover(crossover_points, crossover_plot)
        print(f"Wrote crossover plot: {crossover_plot}")
        print(f"Wrote crossover data: {crossover_csv}")
    else:
        print("Skipped crossover plot: no crossover diagnostics matched the selected points.")

    if missing_voltage and voltage_points:
        print(f"Voltage diagnostics missing at {len(set(missing_voltage))} selected times")
    if missing_crossover and crossover_points:
        print(f"Crossover diagnostics missing at {len(set(missing_crossover))} selected times")
    if not voltage_points and not crossover_points:
        raise SystemExit("No diagnostic plots could be generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
