"""Configure the benchmark and adapt three possible evaluation backends.

``proxies`` evaluates two previously generated scalar ELA proxy functions.
``zdt1`` evaluates an analytic test function for checking optimizer behavior.
``aemec`` calls the existing OpenFOAM voltage-sweep evaluator.

Every backend supplies the same BenchmarkProblem contract: physical input
bounds, two objective names, fixed objective scaling/reference, and a factory
for per-run evaluators. Consequently, switching backends does not change the
budget-accounting or optimizer loop in benchmark.py.

The command defaults to proxies. CFD execution requires selecting aemec and
running without --dry-run in a configured OpenFOAM environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np
import optuna

from .benchmark import INPUTS, BenchmarkProblem, BenchmarkSettings, run_benchmark
from .benchmark_algorithms import ALGORITHMS
from .engine import ObjectiveResult, OptimizationError

# Resolve default assets relative to this repository, regardless of shell cwd.
ROOT = Path(__file__).resolve().parents[2]
# Physical input order: membrane thickness [um], inlet temperature [K], porosity.
# Proxy/AEMEC modes replace these defaults with their recorded study bounds.
LOWER = np.array([10., 298.15, .4])
UPPER = np.array([100., 353.15, .8])


def file_hash(path):
    """Identify the exact proxy/metadata file used, so later edits are detectable."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(value):
    """Parse a quoted CLI command into arguments for the existing process adapter.

    For example, 'make mesh' becomes ('make', 'mesh'). Keeping an argument tuple
    avoids relying on shell parsing when the evaluator launches a subprocess.
    """
    parts = tuple(shlex.split(value))
    if not parts:
        raise argparse.ArgumentTypeError('Command cannot be empty')
    return parts


def build_parser():
    """Expose study controls, backend inputs, and the existing CFD settings.

    --budget/--initial count evaluated designs. --gp-candidates/--gp-restarts
    control work inside the acquisition model and do not increase that budget.
    Objective/reference overrides use raw backend units, never GP-standardized
    coordinates. Backend-specific settings are consumed only by that adapter.
    """
    p = argparse.ArgumentParser(description='Compare GP BO, multiobjective TPE, NSGA-II, and MO-CMA-ES on three inputs/two minimized outputs.')
    p.add_argument('--backend', choices=['proxies', 'zdt1', 'aemec'], default='proxies')
    p.add_argument('--output', type=Path, required=True, help='Fresh benchmark directory')
    p.add_argument('--budget', type=int, default=64, help='Attempt budget per algorithm/seed, including initial designs')
    p.add_argument('--initial', type=int, default=16, help='Shared Sobol count; NSGA-II population size; successful designs seed MO-CMA-ES')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    p.add_argument('--algorithms', choices=ALGORITHMS, nargs='+', default=list(ALGORITHMS))
    # GP computation effort can be varied independently of the expensive budget.
    p.add_argument('--gp-candidates', type=int, default=2048)
    p.add_argument('--gp-restarts', type=int, default=1)
    p.add_argument('--cma-sigma', type=float, default=.2,
                   help='MO-CMA-ES initial Gaussian step size in normalized inputs (default: 0.2)')
    # Keep these common to all runs. Fitting scales separately after each run
    # would make its hypervolume incomparable with the competing algorithms.
    p.add_argument('--reference', type=float, nargs=2, help='Fixed hypervolume reference in raw objective units')
    p.add_argument('--objective-lower', type=float, nargs=2, help='Fixed normalization lower endpoints (both objectives)')
    p.add_argument('--objective-upper', type=float, nargs=2, help='Fixed normalization upper endpoints (both objectives)')
    # ELA artifacts identify the two scalar functions making up the proxy pair.
    p.add_argument('--proxy-root', type=Path, default=ROOT/'ELA/results/proxies_verified')
    p.add_argument('--proxy-rank', type=int, default=1, help='Pair this rank from voltage and crossover proxy folders')
    # OpenFOAM uses the same case adapter and stability criteria as the existing
    # optimizer. The analysis metadata contributes bounds/scales, not training data.
    p.add_argument('--analysis', type=Path, default=ROOT/'ELA/results/analysis', help='AEMEC bounds/objective scales from metadata.json')
    p.add_argument('--case', type=Path, default=ROOT/'run/AEMEC')
    p.add_argument('--work-dir', type=Path, default=ROOT/'opt/work/benchmarks')
    p.add_argument('--mesh-command', type=command, default=('make', 'mesh'))
    p.add_argument('--solver-command', type=command, default=('openFuelCell',))
    p.add_argument('--timeout-minutes', type=float)
    p.add_argument('--target-current-density-a-m2', type=float, default=10000.)
    p.add_argument('--ptl-side', choices=['anode', 'cathode', 'both'], default='anode')
    p.add_argument('--stability-samples', type=int, default=5)
    p.add_argument('--current-relative-tolerance', type=float, default=.05)
    p.add_argument('--voltage-stability-tolerance-v', type=float, default=.005)
    p.add_argument('--crossover-stability-relative-tolerance', type=float, default=.02)
    p.add_argument('--dry-run', action='store_true', help='Validate configuration and print evaluation counts without running algorithms/CFD')
    return p


def build_problem(args):
    """Construct a backend without evaluating any designs or fitting any models.

    Each nested factory closes over its backend configuration and returns
    evaluate(number, physical_x) -> ObjectiveResult. The folder argument gives
    OpenFOAM an isolated place for logs; cheap analytical evaluators need no files.
    Common scaling overrides are applied only after backend defaults are known.
    """
    lower, upper = LOWER.copy(), UPPER.copy()
    metadata = {'backend': args.backend}
    if args.backend == 'proxies':
        from ELA.proxy_functions import ProxyFunction
        if args.proxy_rank < 1:
            raise ValueError('proxy-rank must be positive')
        paths = [args.proxy_root/name/f'proxy_{args.proxy_rank:02d}'/'proxy.json'
                 for name in ['cell_voltage_v', 'crossover_rate_mol_s']]
        # ProxyFunction validates a serialized expression tree and handles its
        # own physical-input normalization. It evaluates a mathematical function,
        # not a fitted GP and not a new high-fidelity CFD observation.
        models = [ProxyFunction(path) for path in paths]
        expected = ['cell_voltage_v', 'crossover_rate_mol_s']
        # Objective order must be stable: column 0 is voltage-like, column 1 is
        # crossover-like. Equal input domains make this a single well-defined
        # two-objective search space rather than two incompatible scalar domains.
        for model, name in zip(models, expected):
            if model.input_columns != INPUTS or model.landscape != name:
                raise ValueError('Proxy objective roles/input columns do not match the benchmark')
        if not np.array_equal(models[0].lower, models[1].lower) or not np.array_equal(models[0].upper, models[1].upper):
            raise ValueError('Proxy domains differ; both objectives must use identical bounds')
        lower, upper = models[0].lower, models[0].upper
        # The proxy artifacts already return dimensionless normalized outputs.
        # Their [0,1] training-pool range is not a guaranteed global range; new
        # designs can return negative values or values above one.
        objective_lower, objective_upper = np.zeros(2), np.ones(2)
        names = ('voltage_proxy', 'crossover_proxy')
        metadata.update({'models': [{'path': str(path.resolve()), 'sha256': file_hash(path)} for path in paths],
                         'warning': 'Independent scalar proxies do not establish a faithful AEMEC joint landscape.'})
        def factory(folder):
            """Wrap the ordered proxy pair in the evaluator's standard result type."""
            return lambda number, x: ObjectiveResult(tuple(model(x) for model in models), {})
    elif args.backend == 'zdt1':
        # Standard ZDT1 has f1=u0 and f2=g*(1-sqrt(u0/g)), where
        # g=1+9*mean(u1,u2). On the unit input box, f1 lies in [0,1] and f2
        # in [0,10]. The true front occurs at u1=u2=0 and is f2=1-sqrt(f1).
        objective_lower, objective_upper = np.zeros(2), np.array([1., 10.])
        names = ('zdt1_f1', 'zdt1_f2')
        metadata['description'] = 'Three-dimensional analytic ZDT1 mapped onto the physical input box; not AEMEC data.'
        def factory(folder):
            """Provide a cheap deterministic evaluator for algorithm smoke checks."""
            def evaluate(number, x):
                # Keep the same physical input interface as AEMEC, then map it
                # back to the conventional unit-box coordinates used by ZDT1.
                u = (x-lower)/(upper-lower)
                g = 1+9*np.mean(u[1:])
                return ObjectiveResult((float(u[0]), float(g*(1-np.sqrt(u[0]/g)))), {})
            return evaluate
    else:
        # The parser limits the third branch to aemec. Importing its adapter here
        # keeps CFD configuration/execution concerns out of the optimizer classes.
        from .case import AemecEvaluationConfig, AemecOpenFoamEvaluator, case_fingerprint, validate_path_layout
        path = args.analysis/'metadata.json'
        analysis = json.loads(path.read_text())
        # Read metadata only. The 128/256 previously evaluated response rows are
        # not handed to any optimizer as free extra initialization observations.
        if analysis.get('input_columns') != INPUTS or analysis.get('objective_columns') != ['cell_voltage_v', 'crossover_rate_mol_s']:
            raise ValueError('AEMEC analysis metadata has incompatible inputs/objectives')
        lower, upper = np.asarray(analysis['input_lower']), np.asarray(analysis['input_upper'])
        objective_lower, objective_upper = np.asarray(analysis['objective_limits'], dtype=float)
        names = ('cell_voltage_v', 'crossover_rate_mol_s')
        # Each callback obtains the existing two objectives at the target current
        # density from a complete voltage sweep, using the shared stability checks.
        config = AemecEvaluationConfig(
            target_current_density_a_m2=args.target_current_density_a_m2, ptl_side=args.ptl_side,
            stability_samples=args.stability_samples, current_relative_tolerance=args.current_relative_tolerance,
            voltage_stability_tolerance_v=args.voltage_stability_tolerance_v,
            crossover_stability_relative_tolerance=args.crossover_stability_relative_tolerance)
        config.validate()
        if args.timeout_minutes is not None and (not np.isfinite(args.timeout_minutes) or args.timeout_minutes <= 0):
            raise ValueError('timeout-minutes must be finite and positive')
        # Namespace scratch cases by the benchmark output location, then by
        # seed/algorithm below. Work and output trees must not overlap each other
        # or the source case: the adapter recreates its scratch case per attempt.
        work_root = args.work_dir/('benchmark-'+hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()[:12])
        validate_path_layout(args.case, work_root, args.output)
        if not args.dry_run:
            # A dry run can inspect configuration on the Mac without having
            # OpenFOAM executables installed. Actual execution checks them first.
            for cmd in (args.mesh_command, args.solver_command):
                if shutil.which(cmd[0]) is None:
                    raise ValueError(f'Executable not found: {cmd[0]}. Run the aemec backend in your configured OpenFOAM environment.')
        metadata.update({'source_case': str(args.case.resolve()), 'case_sha256': case_fingerprint(args.case),
                         'evaluation': config.resume_settings(), 'mesh_command': list(args.mesh_command),
                         'solver_command': list(args.solver_command), 'work_root': str(work_root.resolve()),
                         'analysis_metadata': str(path.resolve()), 'analysis_sha256': file_hash(path),
                         'note': 'Analysis supplies fixed bounds/scales only; existing response rows do not train any optimizer.'})
        def factory(folder):
            """Create a separate CFD adapter for one seed/algorithm or shared pool."""
            work_case = work_root/folder.relative_to(args.output)/'case'
            validate_path_layout(args.case, work_case, folder)
            evaluator = AemecOpenFoamEvaluator(
                source_case=args.case, work_case=work_case, logs_dir=folder/'cfd_logs',
                mesh_command=args.mesh_command, solver_command=args.solver_command, config=config,
                timeout_s=None if args.timeout_minutes is None else 60*args.timeout_minutes)
            # Preserve the positional adapter order: thickness, temperature,
            # porosity. Its result retains physical responses and solver-log paths.
            return lambda number, x: evaluator.evaluate(number, *map(float, x)).as_objective_result()
    # Apply explicit study-wide overrides consistently to any backend. Requiring
    # both endpoint arrays avoids mixing half of a new scale with old defaults.
    if (args.objective_lower is None) != (args.objective_upper is None):
        raise ValueError('Supply both objective-lower and objective-upper together')
    if args.objective_lower is not None:
        objective_lower, objective_upper = np.asarray(args.objective_lower), np.asarray(args.objective_upper)
    # Default reference sits 10% of the declared range beyond each upper endpoint.
    # This defines the measured region; it does not assert that every possible
    # observation will be better than this point. The runner tracks exceptions.
    reference = (objective_lower+1.1*(objective_upper-objective_lower) if args.reference is None
                 else np.asarray(args.reference))
    problem = BenchmarkProblem(lower, upper, objective_lower, objective_upper, reference, names, metadata, factory)
    problem.validate()
    if args.backend == 'aemec' and (lower[0] <= 0 or lower[1] <= 0 or not 0 < lower[2] < upper[2] < 1):
        raise ValueError('AEMEC bounds require positive thickness/temperature and porosity strictly inside (0, 1)')
    return problem


def main(argv=None):
    """Parse/validate the request, optionally preview costs, then run the benchmark.

    Return conventional process codes: 0 for success/dry-run, 2 for a handled
    configuration/runtime error, and 130 for interruption by the user. The runner
    writes incomplete-run status and checkpoints before propagating interrupts.
    """
    args = build_parser().parse_args(argv)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        settings = BenchmarkSettings(args.budget, args.initial, tuple(args.seeds), tuple(args.algorithms),
                                     args.gp_candidates, args.gp_restarts, args.cma_sigma)
        settings.validate()
        problem = build_problem(args)
        # S seeds each need N common initial calls plus A independent adaptive
        # sequences of B-N calls. Charged effort is S*A*B, actual effort is the
        # expression below. No adaptive cross-algorithm response caching occurs.
        maximum = len(settings.seeds)*(settings.initial+len(settings.algorithms)*(settings.budget-settings.initial))
        print(f'Backend: {args.backend}; {len(settings.algorithms)} algorithms × {len(settings.seeds)} seeds × '
              f'{settings.budget} charged attempts. {maximum} actual evaluator calls with shared initialization.', flush=True)
        print(f'Fixed reference: {problem.reference.tolist()}; outputs: {problem.objective_names}', flush=True)
        if args.dry_run:
            # Configuration and source files have been checked, but no output
            # directory, model fit, proxy evaluation or CFD execution is started.
            return 0
        run_benchmark(problem, settings, args.output, progress=lambda message: print(message, flush=True))
        print(f'Benchmark complete: {args.output / "report.md"}')
    except KeyboardInterrupt:
        print('Benchmark interrupted. Completed evaluations and run manifests were retained.', file=sys.stderr)
        return 130
    except (ValueError, OSError, KeyError, OptimizationError) as exc:
        print(f'Benchmark error: {exc}', file=sys.stderr)
        return 2
    return 0
