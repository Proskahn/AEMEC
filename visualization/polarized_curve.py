#!/usr/bin/env python3
"""Plot a AEMEC polarization curve from an openFuelCell log.

The script reads the potentiostatic scan output from run/AEMEC/log.run and
plots voltage against current density. For stepped voltage scans it keeps the
last reported current at each voltage, which is normally the closest sample to
the converged value for that hold.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_ACTIVE_AREA_CM2 = 0.8

TIME_RE = re.compile(r"^Time\s*=\s*([-+0-9.eE]+)")
IBAR_RE = re.compile(
    r"\bibar:\s*([-+0-9.eE]+)\s+voltage:\s*([-+0-9.eE]+)"
)
BOUNDARY_CURRENT_RE = re.compile(
    r"Controlled boundary current \(A\).*?"
    r"signed\s*=\s*([-+0-9.eE]+).*?"
    r"magnitude\s*=\s*([-+0-9.eE]+).*?"
    r"voltage\s*=\s*([-+0-9.eE]+)"
)


@dataclass(frozen=True)
class Sample:
    time: float | None
    voltage_v: float
    current_a: float | None
    current_density_a_m2: float
    source: str

    @property
    def current_density_a_cm2(self) -> float:
        return self.current_density_a_m2 / 1.0e4


def parse_log(log_path: Path, active_area_cm2: float) -> list[Sample]:
    active_area_m2 = active_area_cm2 * 1.0e-4
    samples: list[Sample] = []
    current_time: float | None = None

    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            time_match = TIME_RE.search(line)
            if time_match:
                current_time = float(time_match.group(1))
                continue

            ibar_match = IBAR_RE.search(line)
            if ibar_match:
                ibar = float(ibar_match.group(1))
                voltage = float(ibar_match.group(2))
                samples.append(
                    Sample(
                        time=current_time,
                        voltage_v=voltage,
                        current_a=ibar * active_area_m2,
                        current_density_a_m2=ibar,
                        source="ibar",
                    )
                )
                continue

            boundary_match = BOUNDARY_CURRENT_RE.search(line)
            if boundary_match:
                signed_current = float(boundary_match.group(1))
                voltage = float(boundary_match.group(3))
                samples.append(
                    Sample(
                        time=current_time,
                        voltage_v=voltage,
                        current_a=signed_current,
                        current_density_a_m2=signed_current / active_area_m2,
                        source="boundary",
                    )
                )

    return samples


def last_sample_per_voltage(
    samples: list[Sample], voltage_precision: int
) -> list[Sample]:
    by_voltage: OrderedDict[float, Sample] = OrderedDict()
    for sample in samples:
        key = round(sample.voltage_v, voltage_precision)
        by_voltage[key] = sample

    return sorted(by_voltage.values(), key=lambda sample: sample.voltage_v)


def write_csv(samples: list[Sample], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "time_s",
                "voltage_v",
                "current_a",
                "current_density_a_m2",
                "current_density_a_cm2",
                "source",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    "" if sample.time is None else sample.time,
                    sample.voltage_v,
                    "" if sample.current_a is None else sample.current_a,
                    sample.current_density_a_m2,
                    sample.current_density_a_cm2,
                    sample.source,
                ]
            )


def plot_curve(samples: list[Sample], output_path: Path, use_signed: bool) -> None:
    if use_signed:
        x_values = [sample.current_density_a_cm2 for sample in samples]
        xlabel = "Current density [A/cm2]"
    else:
        x_values = [abs(sample.current_density_a_cm2) for sample in samples]
        xlabel = "|Current density| [A/cm2]"

    y_values = [sample.voltage_v for sample in samples]

    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    ax.plot(x_values, y_values, marker="o", linewidth=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cell voltage [V]")
    ax.set_title("AEMEC Polarization Curve")
    ax.grid(True, which="major", alpha=0.3)
    ax.ticklabel_format(axis="x", style="plain")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a polarization curve from run/AEMEC/log.run."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("run/AEMEC/log.run"),
        help="Path to openFuelCell log file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("visualization/polarized_curve.png"),
        help="Path for the generated PNG plot.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("visualization/polarized_curve.csv"),
        help="Path for the extracted data table.",
    )
    parser.add_argument(
        "--active-area-cm2",
        type=float,
        default=DEFAULT_ACTIVE_AREA_CM2,
        help="Active area used to convert current to current density.",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Plot every parsed sample instead of the last sample per voltage.",
    )
    parser.add_argument(
        "--signed",
        action="store_true",
        help="Plot signed current density instead of magnitude.",
    )
    parser.add_argument(
        "--voltage-precision",
        type=int,
        default=3,
        help="Decimal places used when grouping voltage holds.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    log_path = args.log.resolve()
    output_path = args.output.resolve()
    csv_path = args.csv.resolve()

    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    samples = parse_log(log_path, args.active_area_cm2)
    if not samples:
        raise SystemExit(f"No current/voltage samples found in: {log_path}")

    curve_samples = (
        samples
        if args.all_samples
        else last_sample_per_voltage(samples, args.voltage_precision)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(curve_samples, csv_path)
    plot_curve(curve_samples, output_path, args.signed)

    print(f"Parsed samples: {len(samples)}")
    print(f"Plotted samples: {len(curve_samples)}")
    print(f"Wrote plot: {output_path}")
    print(f"Wrote data: {csv_path}")


if __name__ == "__main__":
    main()
