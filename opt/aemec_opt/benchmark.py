"""Coordinate a fair repeated-seed comparison of multiobjective optimizers.

Execution order is seed -> algorithm -> evaluation attempt. For each seed, a
common Sobol initialization is evaluated once, then replayed into each optimizer
before its adaptive proposals begin. Adaptive observations remain private to
that optimizer. The total budget includes initialization and failed attempts.

There are three distinct representations to keep track of:
  * Optimizer input: a three-element vector in [0, 1]^3.
  * Evaluator input: thickness [um], inlet temperature [K], and porosity [-].
  * Optimizer objectives/metrics: two values under one fixed affine scaling.
Raw evaluator responses are retained in the CSVs for physical interpretation.

This module owns execution, accounting and reporting. benchmark_algorithms.py
owns proposal strategies; benchmark_metrics.py owns Pareto/HV calculations;
benchmark_cli.py constructs the particular objective function being benchmarked.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .benchmark_algorithms import ALGORITHMS, LABELS, GaussianProcessBO, OptunaAlgorithm, sobol_points
from .benchmark_cmaes import MultiObjectiveCMAES, require_deap
from .benchmark_metrics import hypervolume_2d, pareto_indices
from .engine import OptimizationError

# This ordering is shared by every input array, CSV and evaluator adapter.
INPUTS = ['membrane_thickness_um', 'water_inlet_temperature_k', 'ptl_porosity']


@dataclass
class BenchmarkProblem:
    """Backend-independent definition of a three-input, two-objective problem.

    lower/upper: physical input bounds, each with shape (3,).
    objective_lower/upper: fixed affine scaling endpoints, each with shape (2,).
        These are study settings, not guaranteed global objective extrema.
    reference: hypervolume reference in RAW objective units, shape (2,).
    objective_names: ordered names for the two response columns.
    metadata: JSON-compatible backend/settings information for the run manifest.
    evaluator_factory: callable taking a run folder and returning
        evaluate(attempt_number, physical_x) -> ObjectiveResult.

    A factory gives each seed/algorithm its own scratch and log paths while
    allowing the runner to remain independent of OpenFOAM and proxy internals.
    """
    lower: np.ndarray
    upper: np.ndarray
    objective_lower: np.ndarray
    objective_upper: np.ndarray
    reference: np.ndarray
    objective_names: tuple
    metadata: dict
    evaluator_factory: object

    def validate(self):
        """Coerce numerical settings to arrays and reject inconsistent domains.

        Validation checks the reference against the scaling lower endpoints,
        but cannot prove it is worse than every response the backend may return.
        Points outside its measured region are counted later in the history.
        """
        for key, size in [('lower', 3), ('upper', 3), ('objective_lower', 2),
                          ('objective_upper', 2), ('reference', 2)]:
            value = np.asarray(getattr(self, key), dtype=float)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f'{key} must contain {size} finite values')
            setattr(self, key, value)
        if np.any(self.upper <= self.lower) or np.any(self.objective_upper <= self.objective_lower):
            raise ValueError('Upper bounds/scales must exceed lower bounds/scales')
        if np.any(self.reference <= self.objective_lower):
            raise ValueError('Reference coordinates must exceed objective normalization lower endpoints')
        if len(self.objective_names) != 2 or len(set(self.objective_names)) != 2:
            raise ValueError('Two distinct objective names are required')

    def normalize(self, y):
        """Apply the same objective transform to a pair or an (n, 2) response array.

        y_normalized = (y_raw - lower_endpoint) / (upper_endpoint - lower_endpoint).
        Endpoints never change as observations arrive. Values are not clipped:
        a new observation can legitimately fall below zero or above one.
        """
        return (np.asarray(y)-self.objective_lower)/(self.objective_upper-self.objective_lower)


@dataclass
class BenchmarkSettings:
    """Comparison controls that are shared across backend implementations.

    budget: total attempted evaluations per algorithm and seed.
    initial: common initial-design count; also the NSGA-II population size.
        Successful initial designs form MO-CMA-ES's parent population.
    seeds: paired repetitions, each with its own Sobol design and optimizer RNGs.
    algorithms: requested subset of gp/tpe/nsga2/cmaes, evaluated independently.
    gp_candidates/gp_restarts: internal model/search effort, not extra backend calls.
    cma_sigma: initial Gaussian step size in the normalized input box.
    """
    budget: int = 64
    initial: int = 16
    seeds: tuple = (42, 43, 44, 45, 46)
    algorithms: tuple = ALGORITHMS
    gp_candidates: int = 2048
    gp_restarts: int = 1
    cma_sigma: float = .2

    def validate(self):
        """Require a usable initialization and at least one adaptive attempt."""
        if not 4 <= self.initial < self.budget:
            raise ValueError('Require 4 <= initial < budget; the budget includes initialization')
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError('Provide distinct seeds')
        if any(not isinstance(s, int) or not 0 <= s < 2**32-1 for s in self.seeds):
            raise ValueError('Seeds must be integers in [0, 2**32-2]')
        if not self.algorithms or len(set(self.algorithms)) != len(self.algorithms) or set(self.algorithms)-set(ALGORITHMS):
            raise ValueError(f'Choose distinct algorithms from {ALGORITHMS}')
        if self.gp_candidates < 16 or self.gp_restarts < 0:
            raise ValueError('GP candidates must be >=16 and restarts >=0')
        if not np.isfinite(self.cma_sigma) or self.cma_sigma <= 0:
            raise ValueError('CMA sigma must be finite and positive')


def save_json(path, value):
    """Publish a complete JSON checkpoint by replacing a temporary file.

    allow_nan=False prevents silently writing nonstandard JSON containing NaN
    or infinity. A failure while writing leaves the previous checkpoint intact.
    These files record progress; they do not serialize complete sampler RNG state
    and therefore do not provide exact optimization resume.
    """
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save_csv(path, rows):
    """Replace a CSV snapshot only after all accumulated rows have been written."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def evaluate_once(evaluator, number, x):
    """Attempt one physical design and return a uniform success/failure record.

    The callback returns ObjectiveResult with two finite raw objectives and
    optional metadata such as CFD log paths. Expected numerical, simulator and
    filesystem errors produce values=None; the runner still charges the attempt.
    Other exception types and KeyboardInterrupt propagate to stop the benchmark
    rather than being recorded as ordinary failed evaluations.
    """
    started = time.perf_counter()
    try:
        result = evaluator(number, np.asarray(x))
        # Reject incorrect objective count and NaN/Inf before observations can
        # enter any GP, density model, evolutionary population or Pareto metric.
        result.validate(2)
        y = np.asarray(result.values, dtype=float)
        return {'values': y.tolist(), 'status': 'complete', 'error': '',
                'metadata': dict(result.metadata), 'evaluation_s': time.perf_counter()-started}
    except (OptimizationError, ValueError, ArithmeticError, OSError) as exc:
        return {'values': None, 'status': 'failed', 'error': str(exc), 'metadata': {},
                'evaluation_s': time.perf_counter()-started}


def run_benchmark(problem, settings, destination, progress=print):
    """Execute all paired runs and return a DataFrame with one summary per run.

    A completed comparison with S seeds, A algorithms, budget B and initialization
    N charges S*A*B attempts to algorithms but makes S*(N + A*(B-N)) backend
    calls. The difference is the shared initialization, whose values, failures
    and measured evaluation costs are replayed equally to every algorithm.

    All observations/checkpoints go to a fresh destination. Per-run outputs are
    updated after each completed attempt, and status is saved on interruption.
    Existing results are never used to warm-start a new benchmark implicitly.
    """
    problem.validate()
    settings.validate()
    if 'cmaes' in settings.algorithms:
        # Check the new dependency before creating outputs or spending any CFD
        # calls. Runs selecting only the older algorithms do not require DEAP.
        require_deap()
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f'Output already exists: {destination}; select a new directory')
    destination.mkdir(parents=True)
    # The reference must use exactly the same coordinates as the normalized
    # observations passed to hypervolume and GP acquisition. It stays fixed for
    # every seed, every algorithm, and every point in a convergence curve.
    reference = problem.normalize(problem.reference)
    # Record the scientific protocol and numerical-library versions, not just
    # final scores. Source/artifact hashes are supplied in problem.metadata.
    manifest = {'schema_version': 1, 'status': 'running', 'settings': asdict(settings),
                'problem': problem.metadata, 'input_columns': INPUTS,
                'input_lower': problem.lower.tolist(), 'input_upper': problem.upper.tolist(),
                'objective_names': list(problem.objective_names), 'directions': ['minimize', 'minimize'],
                'objective_lower': problem.objective_lower.tolist(), 'objective_upper': problem.objective_upper.tolist(),
                'reference_raw': problem.reference.tolist(), 'reference_normalized': reference.tolist(),
                'budget_policy': 'attempts, including shared initialization, failures and duplicate proposals',
                'shared_initialization': 'evaluated once per seed, charged in full to every algorithm',
                'completed_runs': [], 'actual_evaluator_calls': 0,
                'versions': {p: importlib.metadata.version(p) for p in
                             ['numpy', 'scipy', 'pandas', 'scikit-learn', 'optuna', 'matplotlib']
                             + (['deap'] if 'cmaes' in settings.algorithms else [])}}
    if 'cmaes' in settings.algorithms:
        manifest['cmaes_protocol'] = {
            'implementation': 'DEAP StrategyMultiObjective', 'offspring_per_update': 1,
            'parents': 'successful shared initial designs; first successful fallback if none',
            'initial_sigma': settings.cma_sigma, 'boundary_handling': 'reflection in normalized input box',
            'selection': 'Pareto rank then hypervolume contribution using the fixed benchmark reference',
            'failures': 'charged, excluded from selection, no covariance or step-size update'}
    save_json(destination/'manifest.json', manifest)
    summaries, all_histories, all_pareto = [], [], []
    started = time.perf_counter()
    try:
        for seed in settings.seeds:
            # A repetition gets a fresh initial design and entirely new optimizer
            # instances. Nothing learned under an earlier seed is carried over.
            seed_root = destination/f'seed_{seed}'
            seed_root.mkdir()
            initial = sobol_points(settings.initial, seed)
            shared_folder = seed_root/'shared_initial'
            shared_folder.mkdir()
            shared_evaluator = problem.evaluator_factory(shared_folder)
            shared, shared_rows = [], []
            progress(f'Seed {seed}: evaluating {settings.initial} shared initial designs')
            for i, point in enumerate(initial):
                # Optimizers share unit-box coordinates. The evaluator always
                # receives the corresponding physical design in the INPUTS order.
                physical = problem.lower+point*(problem.upper-problem.lower)
                outcome = evaluate_once(shared_evaluator, i, physical)
                manifest['actual_evaluator_calls'] += 1
                shared.append(outcome)
                # Store failed initial attempts too. All algorithms must see the
                # same initialization outcomes, not different replacement points.
                shared_rows.append({**dict(zip(INPUTS, physical)), 'evaluation': i+1,
                                    **{k: outcome[k] for k in ('status', 'error', 'evaluation_s')},
                                    **dict(zip(problem.objective_names, outcome['values'] or [np.nan]*2))})
                save_csv(shared_folder/'evaluations.csv', shared_rows)
                save_json(shared_folder/'details.json', shared)
                save_json(destination/'manifest.json', manifest)
            for algorithm in settings.algorithms:
                out = seed_root/algorithm
                out.mkdir()
                evaluator = problem.evaluator_factory(out)
                # Creating a fresh optimizer/study here prevents information from
                # an earlier algorithm from leaking into a later one.
                if algorithm == 'gp':
                    optimizer = GaussianProcessBO(seed, reference, settings.gp_candidates, settings.gp_restarts)
                elif algorithm == 'cmaes':
                    optimizer = MultiObjectiveCMAES(seed, reference, settings.cma_sigma)
                else:
                    optimizer = OptunaAlgorithm(algorithm, settings.initial, seed)
                rows, histories, detailed, successful = [], [], [], []
                optimizer_time, evaluation_time = 0., 0.
                for i in range(settings.budget):
                    # Initial outcomes were computed above, but reveal them to
                    # each optimizer one at a time through the normal ask/tell
                    # interface. Only after N attempts may it select new designs.
                    fixed = initial[i] if i < settings.initial else None
                    tick = time.perf_counter()
                    point, proposal = optimizer.ask(fixed)
                    ask_time = time.perf_counter()-tick
                    if point.shape != (3,) or not np.isfinite(point).all() or np.any(point < 0) or np.any(point > 1):
                        raise ValueError(f'{algorithm} proposed an invalid normalized design')
                    if fixed is not None and not np.array_equal(point, fixed):
                        # Treat an altered initial point as an implementation
                        # error: it would violate the paired comparison protocol.
                        raise ValueError('Algorithm did not use the shared initial design')
                    physical = problem.lower+point*(problem.upper-problem.lower)
                    if fixed is not None:
                        # Reuse the complete outcome, including evaluation time
                        # and failure status, without another physical execution.
                        outcome = shared[i]
                    else:
                        outcome = evaluate_once(evaluator, i, physical)
                        manifest['actual_evaluator_calls'] += 1
                    values = outcome['values']
                    # Preserve raw responses for CSVs; send only normalized
                    # responses to the algorithms. None represents a failure and
                    # is never converted to an arbitrary objective penalty.
                    normalized = None if values is None else problem.normalize(values)
                    tick = time.perf_counter()
                    optimizer.tell(point, normalized)
                    tell_time = time.perf_counter()-tick
                    optimizer_time += ask_time+tell_time
                    # This is CHARGED evaluation time: the common initial cost
                    # appears in every run even though it was physically incurred
                    # once per seed. File writing and reporting are not included.
                    evaluation_time += outcome['evaluation_s']
                    if values is not None:
                        successful.append(values)
                    duplicate = any(np.array_equal(physical, [row[k] for k in INPUTS]) for row in rows)
                    # Log repeated proposals rather than granting free retries.
                    # Adaptive duplicates were evaluated above and consume budget.
                    row = {'algorithm': algorithm, 'seed': seed, 'evaluation': i+1,
                           'phase': 'initial' if fixed is not None else 'adaptive',
                           'status': outcome['status'], 'error': outcome['error'],
                           'shared_initial': fixed is not None, 'duplicate_design': duplicate,
                           'evaluation_s': outcome['evaluation_s'], 'optimizer_s': ask_time+tell_time,
                           **dict(zip(INPUTS, physical)),
                           **dict(zip(problem.objective_names, values or [np.nan]*2))}
                    rows.append(row)
                    detailed.append({'evaluation': i+1, 'proposal': proposal, 'evaluation_metadata': outcome['metadata']})
                    # Using every successful observation makes history an archive
                    # metric, even for NSGA-II, whose current population may retain
                    # only a subset. A failed attempt leaves the front unchanged.
                    normalized_success = problem.normalize(np.asarray(successful).reshape(-1, 2))
                    histories.append({'algorithm': algorithm, 'seed': seed, 'evaluation': i+1,
                                      'hypervolume': hypervolume_2d(normalized_success, reference),
                                      'successful': len(successful), 'failed': i+1-len(successful),
                                      'pareto_size': len(pareto_indices(normalized_success)),
                                      # Some valid responses can be worse than the
                                      # predeclared reference in one objective.
                                      # Track them without adjusting that reference.
                                      'outside_reference': int(np.sum(np.any(normalized_success >= reference, axis=1))),
                                      'optimizer_s': optimizer_time, 'evaluation_s_charged': evaluation_time,
                                      'accounted_total_s': optimizer_time+evaluation_time})
                    save_csv(out/'evaluations.csv', rows)
                    save_csv(out/'history.csv', histories)
                    save_json(out/'details.json', detailed)
                    save_json(destination/'manifest.json', manifest)
                    if i >= settings.initial and ((i+1) % 10 == 0 or i+1 == settings.budget):
                        progress(f'{LABELS[algorithm]}, seed {seed}: {i+1}/{settings.budget}, '
                                 f'HV={histories[-1]["hypervolume"]:.6g}, failed={histories[-1]["failed"]}')
                frame = pd.DataFrame(rows)
                valid = frame.loc[frame.status == 'complete']
                # Positive affine scaling preserves dominance, so raw objectives
                # can identify the same Pareto rows while keeping readable units.
                pareto = valid.iloc[pareto_indices(valid[list(problem.objective_names)].to_numpy())]
                pareto.to_csv(out/'pareto.csv', index=False)
                all_pareto.append(pareto)
                all_histories.extend(histories)
                last = histories[-1]
                # Include HV=0 at attempt 0. Trapezoidal integration divided by
                # the budget rewards finding good sets early, not just at the end.
                # It is an average HV over effort, not a score normalized against
                # an unknown true optimum or an AOCC bound.
                summaries.append({'algorithm': algorithm, 'seed': seed, 'budget': settings.budget,
                                  'final_hypervolume': last['hypervolume'],
                                  'mean_hypervolume_over_budget': float(np.trapz(
                                      [0.]+[h['hypervolume'] for h in histories], dx=1)/settings.budget),
                                  'successful': last['successful'], 'failed': last['failed'],
                                  'pareto_size': last['pareto_size'], 'outside_reference': last['outside_reference'],
                                  'duplicate_designs': int(frame.duplicate_design.sum()),
                                  'optimizer_s': optimizer_time, 'evaluation_s_charged': evaluation_time,
                                  'accounted_total_s': optimizer_time+evaluation_time})
                manifest['completed_runs'].append({'algorithm': algorithm, 'seed': seed})
                save_csv(destination/'run_summary.csv', summaries)
                save_csv(destination/'history.csv', all_histories)
                save_json(destination/'manifest.json', manifest)
        summarize(destination, problem, settings, summaries, all_histories, all_pareto)
        manifest['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as exc:
        # Keep completed run/evaluation artifacts and distinguish a user interrupt
        # from a failed run. Do not present an incomplete comparison as complete.
        manifest['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        manifest['error'] = str(exc)
        raise
    finally:
        # This whole-run wall time includes execution/reporting after manifest
        # setup. It differs from the sum of charged per-algorithm times because
        # initialization is physically shared and reporting has overhead.
        manifest['elapsed_wall_s'] = time.perf_counter()-started
        save_json(destination/'manifest.json', manifest)
    return pd.DataFrame(summaries)


def summarize(destination, problem, settings, summaries, histories, fronts):
    """Write descriptive seed statistics and plots from completed runs only.

    Hypervolume curves use a fixed evaluation-attempt axis, including failures.
    Shading is the empirical interquartile range across seeds, not a confidence
    interval. Paired differences compare algorithms under the same initial design.
    These summaries do not assert statistical significance or true Pareto optima.
    """
    runs, history = pd.DataFrame(summaries), pd.DataFrame(histories)
    metrics = ['final_hypervolume', 'mean_hypervolume_over_budget', 'successful', 'failed',
               'pareto_size', 'optimizer_s', 'evaluation_s_charged', 'accounted_total_s']
    aggregate = runs.groupby('algorithm')[metrics].agg(['mean', 'std', 'median', 'min', 'max'])
    # Flatten pandas' two-level column labels into portable CSV field names.
    # Sample standard deviation is undefined (blank) for a single repetition.
    aggregate.columns = ['_'.join(col) for col in aggregate.columns]
    aggregate['runs'] = runs.groupby('algorithm').size()
    aggregate.sort_values('final_hypervolume_mean', ascending=False).to_csv(destination/'algorithm_summary.csv')
    paired = runs.pivot(index='seed', columns='algorithm', values='final_hypervolume')
    for i, a in enumerate(settings.algorithms):
        for b in settings.algorithms[i+1:]:
            # A positive a_minus_b means a achieved larger HV for this same seed.
            paired[f'{a}_minus_{b}'] = paired[a]-paired[b]
    paired.to_csv(destination/'paired_hypervolume.csv')
    import matplotlib
    # Agg supports terminal/headless Linux runs without a graphical display.
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'gp': '#2166ac', 'tpe': '#d95f02', 'nsga2': '#1b9e77', 'cmaes': '#984ea3'}
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for algorithm in settings.algorithms:
        # Compare the algorithms at equal numbers of charged attempts, not at
        # different numbers of successes or generations.
        group = history.loc[history.algorithm == algorithm].groupby('evaluation').hypervolume
        mean, low, high = group.mean(), group.quantile(.25), group.quantile(.75)
        ax.plot(mean.index, mean.to_numpy(), color=colors[algorithm], label=LABELS[algorithm])
        ax.fill_between(mean.index, low.to_numpy(), high.to_numpy(), color=colors[algorithm], alpha=.15)
    ax.axvline(settings.initial, color='gray', ls=':', label='End of shared initialization')
    ax.set(xlabel='Evaluation attempts (including initialization and failures)',
           ylabel='Hypervolume in fixed normalized objective space', title='Mean hypervolume; shading: seed interquartile range')
    ax.legend(fontsize=8)
    fig.savefig(destination/'hypervolume.png', dpi=160)
    plt.close(fig)
    all_fronts = pd.concat(fronts, ignore_index=True)
    # Retain each run's front independently. Pooling then filtering would hide
    # seed variability and could misleadingly resemble a single algorithm run.
    all_fronts.to_csv(destination/'all_run_pareto.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for algorithm in settings.algorithms:
        f = all_fronts.loc[all_fronts.algorithm == algorithm]
        ax.scatter(f[problem.objective_names[0]], f[problem.objective_names[1]], s=20,
                   alpha=.65, color=colors[algorithm], label=LABELS[algorithm])
    ax.set(xlabel=problem.objective_names[0], ylabel=problem.objective_names[1],
           title='Per-run final Pareto sets across seeds (both objectives minimized)')
    ax.legend(fontsize=8)
    fig.savefig(destination/'pareto_fronts.png', dpi=160)
    plt.close(fig)
    (destination/'report.md').write_text(
        '# Multiobjective optimization benchmark\n\n'
        f'Algorithms: {", ".join(LABELS[name] for name in settings.algorithms)}.\n\n'
        f'Backend: {problem.metadata["backend"]}. {len(settings.seeds)} paired seeds, '
        f'{settings.budget} attempts per algorithm/seed, including {settings.initial} shared initial designs.\n\n'
        'Both objectives are minimized. Larger hypervolume is better. The same fixed normalization '
        'and reference point are used in every run; out-of-reference points are recorded, not used '
        'to silently change the reference. mean_hypervolume_over_budget is the trapezoidal area '
        'under the hypervolume history divided by the total attempt budget (not a normalized AOCC).\n\n'
        'algorithm_summary.csv gives descriptive seed statistics; paired_hypervolume.csv gives paired '
        'differences. No statistical significance or true Pareto-optimality claim is made.\n\n'
        'Shared initial observations are evaluated once per seed and charged to each algorithm. '
        'Every subsequent proposal consumes an attempt, including failures and duplicates. Failed '
        'evaluations are excluded from objective models/fronts, never replaced with synthetic penalties.\n\n'
        'optimizer_s measures proposal and update time; evaluation_s_charged charges the shared '
        'initial cost to every run. accounted_total_s is their sum, excluding reporting/I/O. The manifest '
        'records actual simulator calls and whole-benchmark wall time.\n\n'
        'GP BO uses independent Matérn GPs and analytic single-point EHVI over a finite Sobol candidate '
        'pool plus corners/center. It is not noise-integrated NEHVI. TPE uses multivariate MOTPE; '
        'NSGA-II uses Optuna SBX and per-variable random-reset mutation, with population size equal '
        'to the initialization count. No final generation is evaluated past the attempt budget.\n\n'
        + ('MO-CMA-ES uses DEAP StrategyMultiObjective, one offspring per update, and successful shared '
           'initial designs as parents. Selection uses Pareto rank and fixed-reference hypervolume '
           'contribution. Gaussian steps are reflected into the unit input box, and repaired positions '
           'are used for adaptation. Failed attempts skip adaptation; if initialization has no success, '
           'a logged Sobol fallback runs until the first success.\n\n' if 'cmaes' in settings.algorithms else '')
        +
        'For proxy backends, this compares algorithms on an artificial pair of independently '
        'generated scalar functions. Their joint trade-off/Pareto front is not validated against AEMEC; '
        'algorithm rankings need validation on the real model.\n')
