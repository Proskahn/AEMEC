"""Figures for crossover flux density over a galvanostatic current sweep."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .current_sweep import CurrentSweepPoint, crossover_flux_density, target_a_cm2
from .parsing import ExperimentSample
from .project import OptimizationError


def _style(axis: plt.Axes) -> None:
    axis.grid(True, alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)


def plot_current_sweep(path: Path, points: Sequence[CurrentSweepPoint]) -> None:
    if not points:
        raise OptimizationError("No accepted current-sweep points are available to plot")
    ordered = sorted(points, key=lambda point: point.target_current_density_a_cm2)
    current = [point.target_current_density_a_cm2 for point in ordered]
    figure, axis = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    axis.errorbar(
        current,
        [point.mean_crossover_flux_density_mol_m2_s for point in ordered],
        yerr=[point.crossover_flux_density_std_mol_m2_s for point in ordered],
        marker="o",
        linewidth=1.8,
        capsize=3,
        color="#1769aa",
    )
    axis.set_xlabel(r"Current density [A cm$^{-2}$]")
    axis.set_ylabel(r"H$_2$ crossover flux [mol m$^{-2}$ s$^{-1}$]")
    _style(axis)
    figure.suptitle("Hydrogen crossover from 0 to 2 A/cm²")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_current_sweep_timeseries(
    path: Path,
    samples: Sequence[ExperimentSample],
    membrane_area_m2: float,
) -> None:
    if not samples:
        raise OptimizationError("No current-sweep time-series samples are available to plot")
    time = [sample.time_s for sample in samples]
    targets = [target_a_cm2(sample) for sample in samples]
    figure, axes = plt.subplots(2, 1, figsize=(9.2, 6.6), sharex=True, constrained_layout=True)
    axes[0].step(time, targets, where="post", color="0.25", linestyle="--", label="Target")
    axes[0].plot(
        time,
        [sample.current_density_magnitude_a_cm2 for sample in samples],
        color="#1769aa",
        linewidth=1.2,
        label="Measured",
    )
    axes[0].set_ylabel(r"Current density [A cm$^{-2}$]")
    axes[0].legend()
    axes[1].plot(
        time,
        [crossover_flux_density(sample, membrane_area_m2) for sample in samples],
        color="#b54a3a",
        linewidth=1.3,
    )
    axes[1].set_ylabel(r"H$_2$ crossover flux [mol m$^{-2}$ s$^{-1}$]")
    axes[1].set_xlabel("Time [s]")
    for axis in axes:
        _style(axis)
    figure.suptitle("Current-sweep transient history")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def write_current_sweep_reports(
    output_dir: Path,
    samples: Sequence[ExperimentSample],
    points: Sequence[CurrentSweepPoint],
    membrane_area_m2: float,
) -> None:
    plot_current_sweep(output_dir / "current_sweep.png", points)
    plot_current_sweep_timeseries(
        output_dir / "current_sweep_timeseries.png", samples, membrane_area_m2
    )
