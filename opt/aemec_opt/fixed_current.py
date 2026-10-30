"""One galvanostatic CFD run per design, with checked final-window means.

The OpenFOAM clock schedules convergence iterations in this localEuler case;
these durations are not a claim of physically resolved transient behaviour.
The controller adjusts voltage to the requested negative electrolysis current.
We independently check the paired post-solve current, voltage and crossover
over the whole averaging window, rather than trusting current stability alone.
"""
from __future__ import annotations

import csv
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, pstdev

from .case import (
    AemecEvaluationConfig, FATAL_OUTPUT_RE, NORMAL_END_RE, SweepSample,
    _format_number, _named_block_span, _scalar_entry,
    _set_galvanostatic_active, parse_voltage_sweep_samples,
)
from .engine import ObjectiveResult, OptimizationError


@dataclass(frozen=True)
class FixedCurrentRun:
    end_s: float
    delta_t_s: float
    outer_iterations: int


@dataclass(frozen=True)
class FixedCurrentObjective:
    target_current_density_a_m2: float
    cell_voltage_v: float
    crossover_rate_mol_s: float
    window_start_s: float
    window_end_s: float
    window_sample_count: int
    mean_current_density_a_m2: float
    max_target_current_relative_error: float
    current_relative_range: float
    voltage_range_v: float
    crossover_relative_range: float
    voltage_std_v: float
    crossover_std_mol_s: float


@dataclass(frozen=True)
class FixedCurrentEvaluation:
    objective: FixedCurrentObjective
    run: FixedCurrentRun
    solver_log: Path
    mesh_log: Path
    timeseries_csv: Path
    timeseries_plot: Path
    duration_s: float
    ptl_side: str

    def as_objective_result(self) -> ObjectiveResult:
        metadata = asdict(self.objective)
        metadata.pop('cell_voltage_v')
        metadata.pop('crossover_rate_mol_s')
        metadata.update(
            evaluation_mode='fixed-current', ptl_side=self.ptl_side,
            simulation_duration_s=self.run.end_s,
            solver_clock_step=self.run.delta_t_s,
            solver_outer_iterations=self.run.outer_iterations,
            duration_s=self.duration_s,
            solver_log=str(self.solver_log), mesh_log=str(self.mesh_log),
            timeseries_csv=str(self.timeseries_csv), timeseries_plot=str(self.timeseries_plot))
        return ObjectiveResult(
            (self.objective.cell_voltage_v, self.objective.crossover_rate_mol_s), metadata)


def _entry(text: str, name: str, value: str, *, required: bool = True) -> str:
    """Replace a single dictionary entry in an already scoped block."""
    text, count = re.subn(
        rf'(?m)^(\s*{re.escape(name)}\s+)[^;\n]+;',
        lambda match: match.group(1) + value + ';', text)
    if count != 1 and (required or count):
        raise OptimizationError(f'Expected one {name!r} entry in fixed-current settings')
    return text


def configure_fixed_current(case: Path, config: AemecEvaluationConfig) -> FixedCurrentRun:
    """Configure both the meshing template and active dictionary in a scratch case.

    Keep the source time step, initialization, voltage limits and controller
    gain. Setting only endTime would leave a truncated voltage sweep; here both
    galvanostatic control and its single current target are explicitly set.
    """
    config.validate()
    control_path = case / 'system/controlDict.run'
    delta_t = float(_scalar_entry(control_path.read_text(), 'deltaT').group(2))
    if delta_t <= 0:
        raise OptimizationError('Fixed-current deltaT must be positive')
    for name, value in (('duration', config.simulation_duration_s),
                        ('final window', config.final_window_s)):
        steps = value / delta_t
        if not math.isfinite(steps) or not math.isclose(steps, round(steps), rel_tol=0., abs_tol=1e-7):
            raise OptimizationError(f'Fixed-current {name} must be a multiple of deltaT={delta_t:g}')
    window_steps = round(config.final_window_s / delta_t)
    if window_steps < 1 or window_steps + 1 < max(2, config.stability_samples):
        raise OptimizationError('Fixed-current final window has too few time steps for stability checks')

    region_path = case / 'constant/phiEAnode/regionProperties'
    text = _set_galvanostatic_active(region_path.read_text(), True)
    start, end = _named_block_span(text, 'galvanostatic')
    block = text[start:end]
    ibar_start, ibar_end = _named_block_span(block, 'ibar')
    target = _format_number(-config.target_current_density_a_m2)
    duration = _format_number(config.simulation_duration_s)
    block = block[:ibar_start] + (
        f'\n        type table;\n        values ((0 {target}) ({duration} {target}));\n    '
    ) + block[ibar_end:]
    curve_start, curve_end = _named_block_span(block, 'polarizationCurve')
    curve = block[curve_start:curve_end]
    for key, value in (
        ('active', 'true'), ('targets', f'({target})'),
        ('minimumHoldDuration', duration),
        ('targetCurrentTolerance', _format_number(config.target_current_relative_tolerance)),
        ('currentStabilityTolerance', _format_number(config.current_relative_tolerance)),
        ('voltageTolerance', _format_number(config.voltage_stability_tolerance_v)),
        ('stabilitySamples', str(config.stability_samples)),
    ):
        curve = _entry(curve, key, value)
    block = block[:curve_start] + curve + block[curve_end:]
    updates = {region_path: text[:start] + block + text[end:]}
    for name in ('controlDict.run', 'controlDict'):
        path = case / 'system' / name
        if not path.is_file():
            continue
        text = path.read_text()
        for key, value in (('startFrom', 'startTime'), ('startTime', '0'),
                           ('stopAt', 'endTime'), ('endTime', duration),
                           ('deltaT', _format_number(delta_t)),
                           ('writeControl', 'runTime'), ('writeInterval', duration)):
            text = _entry(text, key, value)
        text = _entry(text, 'adjustTimeStep', 'no', required=False)
        text = _entry(text, 'purgeWrite', '1', required=False)
        updates[path] = text
    for path, text in updates.items():
        path.write_text(text, encoding='utf-8')
    return FixedCurrentRun(config.simulation_duration_s, delta_t,
                           round(config.simulation_duration_s / delta_t))


def fixed_current_objective(
    log_text: str, config: AemecEvaluationConfig, run: FixedCurrentRun,
) -> FixedCurrentObjective:
    """Reject incomplete, off-target or drifting endpoints; average valid data.

    The inclusive [end-window, end] interval must contain every expected paired
    sample. Target checks use signed electrolysis current, so a stable wrong
    current or reversed current cannot pass. Range tests catch accumulated
    drift that could be missed by checking only adjacent iterations.
    """
    config.validate()
    if not NORMAL_END_RE.search(log_text) or FATAL_OUTPUT_RE.search(log_text):
        raise OptimizationError('Fixed-current solver did not finish normally')
    samples = parse_voltage_sweep_samples(log_text)
    tolerance = max(1e-9, run.delta_t_s * 1e-6)
    if not samples or abs(samples[-1].time_s - run.end_s) > tolerance:
        raise OptimizationError(f'Fixed-current final sample must be at {run.end_s:g}')
    start = run.end_s - config.final_window_s
    window = [sample for sample in samples if sample.time_s >= start - tolerance]
    expected = round(config.final_window_s / run.delta_t_s) + 1
    if len(window) != expected or len(window) < max(2, config.stability_samples):
        raise OptimizationError('Fixed-current final window has missing or duplicate paired samples')
    if any(abs(sample.time_s - (start + index * run.delta_t_s)) > tolerance
           for index, sample in enumerate(window)):
        raise OptimizationError('Fixed-current final-window samples are not consecutive')
    if any(sample.cell_voltage_v <= 0 or sample.crossover_rate_mol_s < 0 for sample in window):
        raise OptimizationError('Fixed-current final window has nonphysical objectives')

    target = config.target_current_density_a_m2
    currents = [sample.current_density_a_m2 for sample in window]
    target_error = max(abs(value + target) / target for value in currents)
    current_range = (max(currents) - min(currents)) / target
    voltages = [sample.cell_voltage_v for sample in window]
    crossovers = [sample.crossover_rate_mol_s for sample in window]
    voltage_range = max(voltages) - min(voltages)
    crossover_range = (max(crossovers) - min(crossovers)) / max(max(crossovers), 1e-30)
    for label, value, limit in (
        ('target-current error', target_error, config.target_current_relative_tolerance),
        ('current relative range', current_range, config.current_relative_tolerance),
        ('voltage range [V]', voltage_range, config.voltage_stability_tolerance_v),
        ('crossover relative range', crossover_range, config.crossover_stability_relative_tolerance),
    ):
        if value > limit + 1e-12:
            raise OptimizationError(
                f'Fixed-current {label}={value:.6g} exceeds {limit:.6g} over the final '
                f'{config.final_window_s:g} clock units; no objectives accepted. '
                'Inspect the time series and consider a longer simulation duration.')
    return FixedCurrentObjective(
        target, fmean(voltages), fmean(crossovers), start, run.end_s, len(window),
        fmean(currents), target_error, current_range, voltage_range, crossover_range,
        pstdev(voltages), pstdev(crossovers))


def write_fixed_current_timeseries(
    samples: list[SweepSample], csv_path: Path, plot_path: Path,
    config: AemecEvaluationConfig, title: str,
) -> None:
    """Persist raw paired data even when endpoint validation rejects a run."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(SweepSample.__dataclass_fields__)
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(sample) for sample in samples)
    if not samples:
        return
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 8), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 1, sharex=True)
    times = [sample.time_s for sample in samples]
    values = ([sample.current_density_a_m2 / 1e4 for sample in samples],
              [sample.cell_voltage_v for sample in samples],
              [sample.crossover_rate_mol_s for sample in samples])
    for axis, data, label in zip(axes, values, ('Signed current [A/cm²]', 'Cell voltage [V]', 'H₂ crossover [mol/s]')):
        axis.plot(times, data)
        axis.set_ylabel(label)
        axis.axvspan(config.simulation_duration_s - config.final_window_s,
                     config.simulation_duration_s, alpha=.15, color='green')
        axis.grid(alpha=.25)
    axes[0].axhline(-config.target_current_density_a_m2 / 1e4, color='black', linestyle='--', label='Target')
    axes[0].legend()
    axes[-1].set_xlabel('Solver clock (localEuler / pseudo-time)')
    figure.suptitle(title + '\nFixed current; shaded interval = objective averaging window')
    figure.savefig(plot_path, dpi=150)


def finish_fixed_current(log_text, config, run, solver_log, mesh_log,
                         csv_path, plot_path, title, started) -> FixedCurrentEvaluation:
    # Write diagnostics first so rejected trials still have inspectable data.
    write_fixed_current_timeseries(
        parse_voltage_sweep_samples(log_text), csv_path, plot_path, config, title)
    objective = fixed_current_objective(log_text, config, run)
    return FixedCurrentEvaluation(objective, run, solver_log, mesh_log, csv_path,
                                  plot_path, time.monotonic() - started, config.ptl_side)
