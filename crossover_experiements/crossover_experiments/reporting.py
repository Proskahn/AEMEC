"""Study-level CSV and visual reports for fixed-current crossover runs."""

from __future__ import annotations

import csv
from dataclasses import fields
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .parsing import ExperimentSample, ExperimentSummary
from .project import OptimizationError


def write_summary_csv(path: Path, summaries: Sequence[ExperimentSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(ExperimentSummary)]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in sorted(summaries, key=lambda item: item.thickness_um):
            writer.writerow(summary.row())


def _style_axes(axes: plt.Axes) -> None:
    axes.grid(True, alpha=0.25)
    axes.spines[["top", "right"]].set_visible(False)


def plot_timeseries(
    path: Path,
    samples_by_thickness: Mapping[float, Sequence[ExperimentSample]],
    target_current_density_a_cm2: float,
) -> None:
    if not samples_by_thickness:
        raise OptimizationError("No completed experiments are available to plot")
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)
    current_ax, voltage_ax, crossover_ax, fraction_ax = axes.flat
    colors = plt.get_cmap("viridis")
    ordered = sorted(samples_by_thickness.items())
    denominator = max(1, len(ordered) - 1)
    for index, (thickness, samples) in enumerate(ordered):
        color = colors(index / denominator)
        time = [sample.time_s for sample in samples]
        label = f"{thickness:g} µm"
        current_ax.plot(time, [sample.current_density_magnitude_a_cm2 for sample in samples], color=color, label=label)
        voltage_ax.plot(time, [sample.voltage_v for sample in samples], color=color, label=label)
        crossover_ax.plot(time, [sample.crossover_rate_mol_s for sample in samples], color=color, label=label)
        fraction_ax.plot(time, [sample.crossover_fraction_percent for sample in samples], color=color, label=label)
    current_ax.axhline(target_current_density_a_cm2, color="0.25", linestyle="--", linewidth=1.2, label="Target")
    current_ax.set_ylabel(r"$|j|$ [A cm$^{-2}$]")
    voltage_ax.set_ylabel("Cell voltage [V]")
    crossover_ax.set_ylabel(r"H$_2$ crossover [mol s$^{-1}$]")
    fraction_ax.set_ylabel("Crossover / H₂ production [%]")
    for axis in axes.flat:
        axis.set_xlabel("Time [s]")
        _style_axes(axis)
    current_ax.legend(ncols=2, fontsize=8)
    figure.suptitle(r"AEMEC membrane-thickness crossover study at $1\,\mathrm{A\,cm^{-2}}$")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_thickness_summary(path: Path, summaries: Sequence[ExperimentSummary]) -> None:
    ordered = sorted(summaries, key=lambda item: item.thickness_um)
    if not ordered:
        raise OptimizationError("No completed experiment summaries are available to plot")
    thickness = [item.thickness_um for item in ordered]
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.1), constrained_layout=True)
    specs = (
        (
            [item.mean_crossover_rate_mol_s for item in ordered],
            [item.crossover_rate_std_mol_s for item in ordered],
            r"Mean H$_2$ crossover [mol s$^{-1}$]",
        ),
        (
            [item.mean_crossover_fraction_percent for item in ordered],
            None,
            "Crossover / H₂ production [%]",
        ),
        (
            [item.mean_voltage_v for item in ordered],
            [item.voltage_std_v for item in ordered],
            "Mean cell voltage [V]",
        ),
    )
    for axis, (values, errors, ylabel) in zip(axes, specs):
        axis.errorbar(thickness, values, yerr=errors, marker="o", linewidth=1.8, capsize=3, color="#1665a8")
        axis.set_xlabel("Membrane thickness [µm]")
        axis.set_ylabel(ylabel)
        _style_axes(axis)
    figure.suptitle("Final-window membrane-thickness comparison")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def write_reports(
    output_dir: Path,
    samples_by_thickness: Mapping[float, Sequence[ExperimentSample]],
    summaries: Sequence[ExperimentSummary],
    target_current_density_a_cm2: float,
) -> None:
    write_summary_csv(output_dir / "summary.csv", summaries)
    plot_timeseries(
        output_dir / "timeseries.png",
        samples_by_thickness,
        target_current_density_a_cm2,
    )
    plot_thickness_summary(output_dir / "thickness_summary.png", summaries)

