#!/usr/bin/env python3
"""Extract and plot a 2D polarization curve from an openFuelCell log.

The solver prints post-solve collector lines like:

    Time = 0.2
    Controlled boundary current (A) at interconnect0: signed = -1.2, magnitude = 1.2, voltage = 0.2

For older logs, it can also fall back to diagnostic reaction-current lines:

    Time = 0.2
    Total current (A) at acl: 1.278026
    ibar: 0    voltage: 1.2

This script joins those values by time and plots voltage against current.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TIME_RE = re.compile(rf"^\s*Time\s*=\s*(?P<time>{FLOAT_RE})\s*$")
TOTAL_CURRENT_RE = re.compile(
    rf"^\s*Total current \(A\) at (?P<patch>\w+):\s*(?P<current>{FLOAT_RE})\s*$"
)
CONTROL_RE = re.compile(
    rf"^\s*ibar:\s*(?P<ibar>{FLOAT_RE})\s+voltage:\s*(?P<voltage>{FLOAT_RE})\s*$"
)
CONTROLLED_BOUNDARY_RE = re.compile(
    rf"^\s*Controlled boundary current \(A\) at (?P<patch>\w+):\s*"
    rf"signed\s*=\s*(?P<signed>{FLOAT_RE}),\s*"
    rf"magnitude\s*=\s*(?P<magnitude>{FLOAT_RE}),\s*"
    rf"voltage\s*=\s*(?P<voltage>{FLOAT_RE})\s*$"
)


@dataclass
class Sample:
    time: float
    voltage: float
    current: float | None = None
    ibar: float | None = None
    controlled_current: float | None = None
    controlled_signed_current: float | None = None


def parse_log(log_path: Path, current_patch: str, controlled_patch: str) -> list[Sample]:
    """Parse voltage, ibar, and selected total-current samples from a log."""
    samples: list[Sample] = []
    current_time: float | None = None
    currents_at_time: dict[str, float] = {}

    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            if match := TIME_RE.match(line):
                current_time = float(match.group("time"))
                currents_at_time = {}
                continue

            if current_time is None:
                continue

            if match := TOTAL_CURRENT_RE.match(line):
                currents_at_time[match.group("patch")] = float(match.group("current"))
                continue

            if match := CONTROLLED_BOUNDARY_RE.match(line):
                if match.group("patch") == controlled_patch:
                    samples.append(
                        Sample(
                            time=current_time,
                            voltage=float(match.group("voltage")),
                            controlled_current=float(match.group("magnitude")),
                            controlled_signed_current=float(match.group("signed")),
                        )
                    )
                continue

            if match := CONTROL_RE.match(line):
                samples.append(
                    Sample(
                        time=current_time,
                        voltage=float(match.group("voltage")),
                        current=currents_at_time.get(current_patch),
                        ibar=float(match.group("ibar")),
                    )
                )

    return samples


def usable_samples(samples: list[Sample], x_source: str) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []

    for sample in samples:
        if x_source == "auto":
            current = (
                sample.controlled_current
                if sample.controlled_current is not None
                else sample.current
            )
        elif x_source == "controlled-boundary":
            current = sample.controlled_current
        elif x_source == "controlled-boundary-signed":
            current = sample.controlled_signed_current
        elif x_source == "ibar":
            current = sample.ibar
        else:
            current = sample.current

        if current is not None:
            rows.append((sample.time, current, sample.voltage))

    return rows


def write_csv(rows: list[tuple[float, float, float]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time_s", "current_A", "voltage_V"])
        writer.writerows(rows)


def plot_curve(
    rows: list[tuple[float, float, float]],
    output_path: Path,
    title: str,
    x_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    currents = [row[1] for row in rows]
    voltages = [row[2] for row in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.plot(currents, voltages, marker="o", linewidth=1.8, markersize=5)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Voltage (V)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract current/voltage data from an openFuelCell log and plot a polarization curve."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("run/PEMEC/log.run"),
        help="Path to the solver log to parse. Default: run/PEMEC/log.run",
    )
    parser.add_argument(
        "--current-patch",
        default="acl",
        help="Patch name from 'Total current (A) at <patch>'. Default: acl",
    )
    parser.add_argument(
        "--controlled-patch",
        default="interconnect0",
        help="Patch name from 'Controlled boundary current (A) at <patch>'. Default: interconnect0",
    )
    parser.add_argument(
        "--x-source",
        choices=(
            "auto",
            "controlled-boundary",
            "controlled-boundary-signed",
            "total-current",
            "ibar",
        ),
        default="auto",
        help=(
            "Use post-solve controlled boundary current when available, signed controlled boundary "
            "current, older total reaction current, or ibar. Default: auto"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("visualization/polarized_curve.png"),
        help="Output image path. Default: visualization/polarized_curve.png",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("visualization/polarized_curve.csv"),
        help="Output CSV path. Default: visualization/polarized_curve.csv",
    )
    parser.add_argument(
        "--title",
        default="Polarization Curve",
        help="Plot title. Default: Polarization Curve",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.log.exists():
        raise SystemExit(f"Log file not found: {args.log}")

    samples = parse_log(args.log, args.current_patch, args.controlled_patch)
    x_source = args.x_source
    rows = usable_samples(samples, x_source)

    if not rows:
        if x_source == "auto":
            hint = (
                f"No samples found. For new logs this expects controlled-boundary samples at "
                f"'{args.controlled_patch}'; for old logs it expects total current at "
                f"'{args.current_patch}'."
            )
        elif x_source == "controlled-boundary":
            hint = (
                f"No post-solve controlled-boundary samples found at patch "
                f"'{args.controlled_patch}'. Rebuild/rerun the solver, or use "
                f"--x-source total-current for older logs."
            )
        elif x_source == "controlled-boundary-signed":
            hint = (
                f"No signed controlled-boundary samples found at patch "
                f"'{args.controlled_patch}'. Rebuild/rerun the solver, or use "
                f"--x-source total-current for older logs."
            )
        elif x_source == "total-current":
            hint = f"No samples found with voltage and total current at patch '{args.current_patch}'."
        else:
            hint = "No samples found with voltage and ibar."
        raise SystemExit(f"{hint} Check the log path and current source.")

    x_label = (
        "Current (A)"
        if x_source in ("auto", "controlled-boundary", "total-current")
        else "Signed current (A)"
        if x_source == "controlled-boundary-signed"
        else "Current density ibar (A/m2)"
    )
    write_csv(rows, args.csv)
    plot_curve(rows, args.output, args.title, x_label)

    print(f"Parsed {len(rows)} samples from {args.log}")
    print(f"Wrote CSV: {args.csv}")
    print(f"Wrote plot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
