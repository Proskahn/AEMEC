#!/usr/bin/env python3
"""Compare polarization and H2 crossover for selected membrane thicknesses."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_THICKNESSES_UM = (20.0, 40.0, 60.0, 80.0)


@dataclass(frozen=True)
class TrialCurve:
    trial_number: int
    membrane_thickness_um: float
    current_density_a_cm2: tuple[float, ...]
    cell_voltage_v: tuple[float, ...]
    crossover_rate_mol_s: tuple[float, ...]


def _format_thickness(value: float) -> str:
    return f"{value:g}"


def _trial_rows(results_path: Path) -> list[dict[str, str]]:
    if not results_path.is_file():
        raise ValueError(f"optimization results not found: {results_path}")
    with results_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError(f"optimization results are empty: {results_path}")
    required = {"trial", "membrane_thickness_um"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(
            f"optimization results are missing columns: {', '.join(sorted(missing))}"
        )
    return rows


def _select_trial_rows(
    rows: Sequence[dict[str, str]],
    thicknesses_um: Sequence[float],
    tolerance_um: float,
) -> list[tuple[int, float]]:
    if not thicknesses_um:
        raise ValueError("at least one membrane thickness is required")
    if not math.isfinite(tolerance_um) or tolerance_um < 0:
        raise ValueError("thickness tolerance must be finite and non-negative")

    available: list[tuple[int, float]] = []
    for row in rows:
        try:
            trial = int(row["trial"])
            thickness = float(row["membrane_thickness_um"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid trial or membrane thickness in results CSV") from exc
        if math.isfinite(thickness):
            available.append((trial, thickness))

    selected: list[tuple[int, float]] = []
    for requested in thicknesses_um:
        if not math.isfinite(requested) or requested <= 0:
            raise ValueError("requested membrane thicknesses must be positive")
        matches = [
            item for item in available if abs(item[1] - requested) <= tolerance_um
        ]
        if not matches:
            values = ", ".join(
                _format_thickness(value)
                for value in sorted({value for _, value in available})
            )
            raise ValueError(
                f"no completed trial found at {requested:g} um; "
                f"available thicknesses: {values}"
            )
        selected.append(min(matches, key=lambda item: item[0]))
    return selected


def _load_curve(
    study_dir: Path,
    trial_number: int,
    membrane_thickness_um: float,
) -> TrialCurve:
    curve_path = (
        study_dir
        / "logs"
        / f"trial_{trial_number:04d}_polarization_curve.csv"
    )
    if not curve_path.is_file():
        raise ValueError(
            f"curve CSV not found: {curve_path}; run opt/export_trial_curves.py first"
        )
    with curve_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError(f"curve CSV is empty: {curve_path}")

    current_density: list[float] = []
    voltage: list[float] = []
    crossover_rate: list[float] = []
    for row in rows:
        try:
            current = float(row["final_current_density_magnitude_a_cm2"])
            cell_voltage = float(row["cell_voltage_v"])
            crossover = float(row["final_crossover_rate_mol_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid I-V or crossover data in: {curve_path}") from exc
        if not all(
            math.isfinite(value) for value in (current, cell_voltage, crossover)
        ):
            raise ValueError(f"non-finite I-V or crossover data in: {curve_path}")
        current_density.append(current)
        voltage.append(cell_voltage)
        crossover_rate.append(crossover)

    return TrialCurve(
        trial_number,
        membrane_thickness_um,
        tuple(current_density),
        tuple(voltage),
        tuple(crossover_rate),
    )


def load_selected_curves(
    study_dir: Path,
    thicknesses_um: Sequence[float] = DEFAULT_THICKNESSES_UM,
    tolerance_um: float = 1.0e-6,
) -> list[TrialCurve]:
    study_dir = study_dir.resolve()
    selected = _select_trial_rows(
        _trial_rows(study_dir / "optimization_results.csv"),
        thicknesses_um,
        tolerance_um,
    )
    return [
        _load_curve(study_dir, trial, thickness)
        for trial, thickness in selected
    ]


def plot_curves(curves: Sequence[TrialCurve], output_path: Path) -> None:
    if not curves:
        raise ValueError("no polarization curves were provided")

    markers = ("o", "s", "^", "D", "v", "P", "X")
    figure, (voltage_axes, crossover_axes) = plt.subplots(
        2,
        1,
        figsize=(7.4, 8.0),
        sharex=True,
        constrained_layout=True,
    )
    for index, curve in enumerate(curves):
        voltage_line = voltage_axes.plot(
            curve.current_density_a_cm2,
            curve.cell_voltage_v,
            marker=markers[index % len(markers)],
            linewidth=1.8,
            markersize=5,
            label=f"{curve.membrane_thickness_um:g} µm",
        )[0]
        crossover_axes.plot(
            curve.current_density_a_cm2,
            curve.crossover_rate_mol_s,
            color=voltage_line.get_color(),
            marker=markers[index % len(markers)],
            linewidth=1.8,
            markersize=5,
        )

    figure.suptitle("Membrane-Thickness Performance Comparison")
    voltage_axes.set_ylabel("Cell voltage [V]")
    voltage_axes.set_title("Polarization")
    voltage_axes.grid(True, which="major", alpha=0.3)
    voltage_axes.legend(title="Membrane thickness")

    crossover_axes.set_xlabel(r"$|j|$ [A/cm$^2$]")
    crossover_axes.set_ylabel(r"H$_2$ crossover rate [mol/s]")
    crossover_axes.set_title(r"H$_2$ crossover")
    crossover_axes.grid(True, which="major", alpha=0.3)
    crossover_axes.ticklabel_format(axis="x", style="plain")
    crossover_axes.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study_dir",
        type=Path,
        help="Study directory containing optimization_results.csv and logs/.",
    )
    parser.add_argument(
        "--thicknesses",
        type=float,
        nargs="+",
        default=DEFAULT_THICKNESSES_UM,
        metavar="UM",
        help="Membrane thicknesses to compare (default: 20 40 60 80).",
    )
    parser.add_argument(
        "--thickness-tolerance-um",
        type=float,
        default=1.0e-6,
        help="Absolute tolerance for matching completed trials (default: 1e-6 um).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output PNG path (default: "
            "selected_polarization_and_crossover_curves.png in study)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    study_dir = args.study_dir.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else study_dir / "selected_polarization_and_crossover_curves.png"
    )
    try:
        curves = load_selected_curves(
            study_dir,
            args.thicknesses,
            args.thickness_tolerance_um,
        )
        plot_curves(curves, output_path)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    for curve in curves:
        print(
            f"trial {curve.trial_number}: "
            f"{curve.membrane_thickness_um:g} um, "
            f"{len(curve.cell_voltage_v)} points"
        )
    print(f"Wrote comparison plot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
