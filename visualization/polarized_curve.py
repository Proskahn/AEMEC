#!/usr/bin/env python3
"""Extract and plot an AEMEC polarization curve from an openFuelCell log.

The preferred source is the solver's ``Controlled boundary current`` record:
it contains both the post-solve cell voltage and the actual patch current
density. For a galvanostatic stepped scan, the final record for each
``galvanostatic target`` is retained. Potentiostatic scans are grouped by
their commanded voltage instead.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_ACTIVE_AREA_CM2 = 0.8

FLOAT_PATTERN = r"[-+0-9.eE]+"
TIME_RE = re.compile(rf"^Time\s*=\s*({FLOAT_PATTERN})")
GALVANOSTATIC_TARGET_RE = re.compile(
    rf"\bgalvanostatic\s+target:\s*({FLOAT_PATTERN})\s*A/m2\b",
    re.IGNORECASE,
)
IBAR_RE = re.compile(
    rf"\bibar:\s*({FLOAT_PATTERN})\s+voltage:\s*({FLOAT_PATTERN})"
)
BOUNDARY_CURRENT_RE = re.compile(
    rf"Controlled\s+boundary\s+current\s+\(A\).*?"
    rf"signed\s*=\s*({FLOAT_PATTERN}).*?"
    rf"magnitude\s*=\s*({FLOAT_PATTERN}).*?"
    rf"current\s+density\s*=\s*({FLOAT_PATTERN})\s*A/m2.*?"
    rf"voltage\s*=\s*({FLOAT_PATTERN})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Sample:
    time: float | None
    voltage_v: float
    current_a: float | None
    current_density_a_m2: float
    target_current_density_a_m2: float | None
    source: str

    @property
    def current_density_a_cm2(self) -> float:
        return self.current_density_a_m2 / 1.0e4

    @property
    def target_current_error_a_m2(self) -> float | None:
        if self.target_current_density_a_m2 is None:
            return None
        return self.current_density_a_m2 - self.target_current_density_a_m2

    @property
    def target_current_error_relative(self) -> float | None:
        if self.target_current_density_a_m2 in (None, 0.0):
            return None
        return abs(
            self.target_current_error_a_m2 / self.target_current_density_a_m2
        )


def parse_log(log_path: Path, active_area_cm2: float) -> list[Sample]:
    """Return boundary-current samples, or legacy ``ibar`` samples as fallback.

    ``ibar`` is a controller diagnostic and can precede the post-solve patch
    current. It is only used when a log has no boundary-current records.
    """
    active_area_m2 = active_area_cm2 * 1.0e-4
    boundary_samples: list[Sample] = []
    ibar_samples: list[Sample] = []
    current_time: float | None = None
    target_current_density_a_m2: float | None = None

    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            time_match = TIME_RE.search(line)
            if time_match:
                current_time = float(time_match.group(1))
                continue

            target_match = GALVANOSTATIC_TARGET_RE.search(line)
            if target_match:
                target_current_density_a_m2 = float(target_match.group(1))
                continue

            ibar_match = IBAR_RE.search(line)
            if ibar_match:
                ibar = float(ibar_match.group(1))
                voltage = float(ibar_match.group(2))
                ibar_samples.append(
                    Sample(
                        time=current_time,
                        voltage_v=voltage,
                        current_a=ibar * active_area_m2,
                        current_density_a_m2=ibar,
                        target_current_density_a_m2=target_current_density_a_m2,
                        source="ibar-fallback",
                    )
                )
                continue

            boundary_match = BOUNDARY_CURRENT_RE.search(line)
            if boundary_match:
                signed_current = float(boundary_match.group(1))
                current_density = float(boundary_match.group(3))
                voltage = float(boundary_match.group(4))
                boundary_samples.append(
                    Sample(
                        time=current_time,
                        voltage_v=voltage,
                        current_a=signed_current,
                        current_density_a_m2=current_density,
                        target_current_density_a_m2=target_current_density_a_m2,
                        source="boundary-current-density",
                    )
                )

    return boundary_samples or ibar_samples


def last_sample_per_voltage(
    samples: list[Sample], voltage_precision: int
) -> list[Sample]:
    by_voltage: OrderedDict[float, Sample] = OrderedDict()
    for sample in samples:
        key = round(sample.voltage_v, voltage_precision)
        by_voltage[key] = sample

    return sorted(by_voltage.values(), key=lambda sample: sample.voltage_v)


def last_sample_per_target_current(
    samples: list[Sample], current_precision: int
) -> list[Sample]:
    """Keep the last post-solve sample for each galvanostatic setpoint."""
    if any(sample.target_current_density_a_m2 is None for sample in samples):
        raise ValueError("No galvanostatic target values found")

    by_target: OrderedDict[float, Sample] = OrderedDict()
    for sample in samples:
        assert sample.target_current_density_a_m2 is not None
        key = round(sample.target_current_density_a_m2, current_precision)
        by_target[key] = sample

    return sorted(
        by_target.values(),
        key=lambda sample: abs(sample.target_current_density_a_m2 or 0.0),
    )


def split_target_convergence(
    samples: list[Sample],
    relative_tolerance: float,
    zero_target_absolute_tolerance_a_m2: float,
) -> tuple[list[Sample], list[Sample]]:
    """Split samples by agreement between actual and commanded current.

    A point is only a polarization-curve point once the post-solve boundary
    current has reached the current requested by the galvanostatic controller.
    Untargeted legacy samples are retained because no comparison is possible.
    """
    on_target: list[Sample] = []
    off_target: list[Sample] = []

    for sample in samples:
        target = sample.target_current_density_a_m2
        if target is None:
            on_target.append(sample)
            continue

        error = abs(sample.current_density_a_m2 - target)
        tolerance = (
            zero_target_absolute_tolerance_a_m2
            if target == 0.0
            else abs(target) * relative_tolerance
        )
        (on_target if error <= tolerance else off_target).append(sample)

    return on_target, off_target


def last_sample_per_hold(
    samples: list[Sample], hold_duration: float
) -> list[Sample]:
    """Keep the last sample in each (0, hold], (hold, 2*hold], ... interval."""
    by_hold: OrderedDict[int, Sample] = OrderedDict()
    for sample in samples:
        if sample.time is None:
            continue
        hold_index = max(1, math.ceil(sample.time / hold_duration))
        by_hold[hold_index] = sample

    if not by_hold:
        raise ValueError("Cannot group current holds because the log has no Time entries")

    return list(by_hold.values())


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
                "target_current_density_a_m2",
                "target_current_error_a_m2",
                "target_current_error_relative",
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
                    ""
                    if sample.target_current_density_a_m2 is None
                    else sample.target_current_density_a_m2,
                    ""
                    if sample.target_current_error_a_m2 is None
                    else sample.target_current_error_a_m2,
                    ""
                    if sample.target_current_error_relative is None
                    else sample.target_current_error_relative,
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
        description="Extract a polarization curve from run/AEMEC/log.run."
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
        help="Fallback area for legacy ibar-only logs (default: 0.8 cm2).",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Plot every parsed post-solve sample instead of one per scan point.",
    )
    parser.add_argument(
        "--include-off-target",
        action="store_true",
        help="Include points whose actual current has not reached its target.",
    )
    parser.add_argument(
        "--scan-mode",
        choices=("current", "voltage"),
        default="current",
        help="Controlled quantity used for grouping stepped scan points (default: current).",
    )
    parser.add_argument(
        "--hold-duration",
        type=float,
        default=10.0,
        help="Fallback hold duration when target records are absent (default: 10 s).",
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
    parser.add_argument(
        "--current-precision",
        type=int,
        default=3,
        help="Decimal places used when grouping galvanostatic targets in A/m2.",
    )
    parser.add_argument(
        "--target-relative-tolerance",
        type=float,
        default=0.05,
        help="Maximum relative actual-vs-target current error (default: 0.05).",
    )
    parser.add_argument(
        "--zero-target-absolute-tolerance-a-m2",
        type=float,
        default=100.0,
        help="Maximum |current density| for a zero-current point (default: 100 A/m2).",
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

    if args.hold_duration <= 0:
        raise SystemExit("--hold-duration must be greater than zero")
    if args.active_area_cm2 <= 0:
        raise SystemExit("--active-area-cm2 must be greater than zero")
    if args.current_precision < 0:
        raise SystemExit("--current-precision cannot be negative")
    if not 0 <= args.target_relative_tolerance < 1:
        raise SystemExit("--target-relative-tolerance must be in [0, 1)")
    if args.zero_target_absolute_tolerance_a_m2 < 0:
        raise SystemExit("--zero-target-absolute-tolerance-a-m2 cannot be negative")

    if args.all_samples:
        curve_samples = samples
    elif args.scan_mode == "current":
        try:
            curve_samples = last_sample_per_target_current(
                samples, args.current_precision
            )
        except ValueError:
            curve_samples = last_sample_per_hold(samples, args.hold_duration)
    else:
        curve_samples = last_sample_per_voltage(samples, args.voltage_precision)

    accepted_samples, rejected_samples = split_target_convergence(
        curve_samples,
        args.target_relative_tolerance,
        args.zero_target_absolute_tolerance_a_m2,
    )
    if rejected_samples and not args.include_off_target:
        curve_samples = accepted_samples
    if not curve_samples:
        raise SystemExit(
            "No on-target samples remain. Increase the scan hold duration or use "
            "--include-off-target to inspect the raw controller response."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(curve_samples, csv_path)
    plot_curve(curve_samples, output_path, args.signed)

    print(f"Parsed samples: {len(samples)}")
    print(f"Plotted samples: {len(curve_samples)}")
    if rejected_samples:
        print(
            f"Off-target samples: {len(rejected_samples)} "
            f"({'included' if args.include_off_target else 'excluded'})"
        )
    print(f"Wrote plot: {output_path}")
    print(f"Wrote data: {csv_path}")


if __name__ == "__main__":
    main()
