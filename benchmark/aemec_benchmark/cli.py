"""Configure algorithms and compare their proposals using CFD responses only."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys

import optuna

from .runner import BenchmarkSettings, run_benchmark
from .algorithms import ALGORITHMS, DEFAULT_ALGORITHMS
from .llamea import LLaMEASettings
from .proxy_assets import resolve_proxy_paths, load_proxy_assets
from aemec_opt.engine import OptimizationError

# Resolve default assets relative to this repository, regardless of shell cwd.
ROOT = Path(__file__).resolve().parents[2]


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
    """Expose algorithm controls and the common CFD evaluation settings.

    --budget/--initial count evaluated designs. --gp-candidates/--gp-restarts
    control work inside the acquisition model and do not increase that budget.
    Objective/reference overrides use raw backend units, never GP-standardized
    coordinates. Proxy options configure only the ELA proposal algorithm.
    """
    p = argparse.ArgumentParser(description='Compare GP BO, TPE, NSGA-II, MO-CMA-ES, ELA proxy search, and LLaMEA using CFD responses only.')
    p.add_argument('--backend', choices=['aemec'], default='aemec',
                   help='CFD is the only evaluator; retained for compatibility with existing commands')
    p.add_argument('--output', type=Path, required=True, help='Fresh benchmark directory')
    p.add_argument('--budget', type=int, default=64, help='Attempt budget per algorithm/seed, including initial designs')
    p.add_argument('--initial', type=int, default=16, help='Shared Sobol count and NSGA-II/MO-CMA population size')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    p.add_argument('--algorithms', choices=ALGORITHMS, nargs='+', default=list(DEFAULT_ALGORITHMS))
    # GP computation effort can be varied independently of the expensive budget.
    p.add_argument('--gp-candidates', type=int, default=2048)
    p.add_argument('--gp-restarts', type=int, default=1)
    # Recognize the removed option only to give old scripts an actionable error.
    # MoCmaSampler does not expose an initial-sigma parameter; never ignore it.
    p.add_argument('--cma-sigma', type=float, help=argparse.SUPPRESS)
    # Keep these common to all runs. Fitting scales separately after each run
    # would make its hypervolume incomparable with the competing algorithms.
    p.add_argument('--reference', type=float, nargs=2, help='Fixed hypervolume reference in raw objective units')
    p.add_argument('--objective-lower', type=float, nargs=2, help='Fixed normalization lower endpoints (both objectives)')
    p.add_argument('--objective-upper', type=float, nargs=2, help='Fixed normalization upper endpoints (both objectives)')
    # Used only by --algorithms ela_proxy. Result directory names differ between machines and generation runs. An
    # explicit root or file pair takes precedence; otherwise discover one pair.
    sources = p.add_mutually_exclusive_group()
    sources.add_argument('--proxy-root', type=Path,
                         help='Directory containing cell_voltage_v/ and crossover_rate_mol_s/; '
                              'for ela_proxy; default: discover a unique pair under ELA/results')
    sources.add_argument('--proxy-files', type=Path, nargs=2, metavar=('VOLTAGE_JSON', 'CROSSOVER_JSON'),
                         help='Explicit voltage and crossover proxy JSON files, in that order')
    p.add_argument('--proxy-rank', type=int, default=1,
                   help='Rank to load with --proxy-root or automatic discovery (default: 1)')
    p.add_argument('--proxy-candidates', type=int, default=2048,
                   help='Cheap candidate pool per ela_proxy proposal, not the CFD budget')
    # LLaMEA evolves optimizer code. These controls bound code generation and
    # divide the existing CFD budget; they do not create a second CFD allowance.
    p.add_argument('--llamea-model', help='Required with llamea: exact model name available from your provider')
    p.add_argument('--llamea-provider', choices=['ollama', 'openai'], default='ollama')
    p.add_argument('--llamea-candidates', type=int, default=8, help='Maximum generated optimizer candidates per seed')
    p.add_argument('--llamea-candidate-budget', type=int, default=4,
                   help='CFD attempts per generated optimizer; all are charged to --budget')
    p.add_argument('--llamea-proposal-timeout', type=float, default=10., help='Seconds allowed for one generated proposal')
    p.add_argument('--llamea-request-timeout', type=float, default=120., help='LLM request timeout in seconds; no automatic retries')
    p.add_argument('--llamea-max-tokens', type=int, default=4096, help='Maximum response tokens per LLM request')
    p.add_argument('--llamea-api-key-env', default='OPENAI_API_KEY', help='Environment variable NAME, never the key itself')
    p.add_argument('--llamea-base-url', help='Optional provider base URL, without credentials')
    # A portable study configuration allows CFD runs on a fresh checkout with
    # no ELA/results directory. Existing analysis metadata is an optional source
    # of the same bounds/scales; neither option supplies training observations.
    configuration = p.add_mutually_exclusive_group()
    configuration.add_argument('--problem-config', type=Path,
                               help='AEMEC bounds/scales JSON (default: benchmark/configs/aemec_benchmark.json)')
    configuration.add_argument('--analysis', type=Path,
                               help='Read AEMEC bounds/scales from this directory\'s metadata.json instead')
    p.add_argument('--case', type=Path, default=ROOT/'run/AEMEC')
    p.add_argument('--work-dir', type=Path, default=ROOT/'benchmark/work')
    p.add_argument('--mesh-command', type=command, default=('make', 'mesh'))
    p.add_argument('--solver-command', type=command, default=('openFuelCell',))
    p.add_argument('--timeout-minutes', type=float)
    p.add_argument('--target-current-density-a-m2', type=float, default=10000.)
    from aemec_opt.protocol import add_protocol_arguments
    add_protocol_arguments(p)
    p.add_argument('--ptl-side', choices=['anode', 'cathode', 'both'], default='anode')
    p.add_argument('--stability-samples', type=int, default=5)
    p.add_argument('--current-relative-tolerance', type=float, default=.05)
    p.add_argument('--voltage-stability-tolerance-v', type=float, default=.005)
    p.add_argument('--crossover-stability-relative-tolerance', type=float, default=.02)
    p.add_argument('--dry-run', action='store_true', help='Validate configuration and print evaluation counts without running algorithms/CFD')
    return p


def build_problem(args):
    """Build the CFD evaluator independently of any algorithm's proposal model."""
    from .evaluation import build_cfd_problem
    return build_cfd_problem(args, ROOT)


def main(argv=None):
    """Parse/validate the request, optionally preview costs, then run the benchmark.

    Return conventional process codes: 0 for success/dry-run, 2 for a handled
    configuration/runtime error, and 130 for interruption by the user. The runner
    writes incomplete-run status and checkpoints before propagating interrupts.
    """
    args = build_parser().parse_args(argv)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        if args.cma_sigma is not None:
            raise ValueError('Remove --cma-sigma: OptunaHub MoCmaSampler fixes its initial step size at 1/6')
        llamea_settings = None
        if 'llamea' in args.algorithms or args.llamea_model is not None:
            llamea_settings = LLaMEASettings(
                model=args.llamea_model or '', provider=args.llamea_provider,
                candidates=args.llamea_candidates, candidate_budget=args.llamea_candidate_budget,
                proposal_timeout=args.llamea_proposal_timeout, request_timeout=args.llamea_request_timeout,
                max_tokens=args.llamea_max_tokens, api_key_env=args.llamea_api_key_env,
                base_url=args.llamea_base_url)
        settings = BenchmarkSettings(budget=args.budget, initial=args.initial,
                                     seeds=tuple(args.seeds), algorithms=tuple(args.algorithms),
                                     gp_candidates=args.gp_candidates, gp_restarts=args.gp_restarts,
                                     proxy_candidates=args.proxy_candidates, llamea=llamea_settings)
        settings.validate()
        if 'ela_proxy' not in settings.algorithms and (args.proxy_root is not None or args.proxy_files is not None):
            raise ValueError('Proxy file options require --algorithms ela_proxy; all algorithms are evaluated by CFD')
        problem = build_problem(args)
        assets = None
        if 'ela_proxy' in settings.algorithms:
            assets = load_proxy_assets(resolve_proxy_paths(args), problem.lower, problem.upper)
        # S seeds each need N common initial calls plus A independent adaptive
        # sequences of B-N calls. Charged effort is S*A*B, actual effort is the
        # expression below. No adaptive cross-algorithm response caching occurs.
        maximum = len(settings.seeds)*(settings.initial+len(settings.algorithms)*(settings.budget-settings.initial))
        print(f'Evaluator: CFD (AEMEC/OpenFOAM); {len(settings.algorithms)} algorithms × {len(settings.seeds)} seeds × '
              f'{settings.budget} charged attempts. {maximum} actual evaluator calls with shared initialization.', flush=True)
        print(f'Fixed reference: {problem.reference.tolist()}; outputs: {problem.objective_names}', flush=True)
        if args.backend == 'aemec':
            protocol = ('fixed-current runs '
                        f'({args.simulation_duration_s:g} clock units, final {args.final_window_s:g} averaged)'
                        if args.evaluation_mode == 'fixed-current' else 'voltage sweeps')
            print(f'Actual OpenFOAM {protocol} at {args.target_current_density_a_m2:g} A/m²; '
                  'objective units: V and mol/s. Only CFD responses are scored.', flush=True)
            print(f'Source case: {problem.metadata["source_case"]}', flush=True)
            print(f'Bounds/scales: {problem.metadata["configuration"]["path"]}', flush=True)
        if assets is not None:
            # Show the exact artifacts even on a dry run; the manifest also
            # retains their absolute paths and hashes once execution starts.
            for label, model in zip(('Voltage', 'Crossover'), assets.metadata['models']):
                print(f'ELA proposal model ({label}): {model["path"]}', flush=True)
        if llamea_settings is not None:
            steps = min(llamea_settings.candidate_budget, settings.budget-settings.initial)
            limit = min(llamea_settings.candidates, (settings.budget-settings.initial)//steps)
            print(f'LLaMEA: {llamea_settings.provider}/{llamea_settings.model}; up to {limit} code candidates × '
                  f'{steps} CFD attempts per seed, within the existing budget; then deploy the best. '
                  'LLM requests incur separate provider cost.', flush=True)
        if args.dry_run:
            # Configuration and source files have been checked, but no output
            # directory, model fit, proxy evaluation or CFD execution is started.
            print('Dry run only: no evaluations executed; CFD executables and runtime compatibility were not checked.', flush=True)
            return 0
        run_benchmark(problem, settings, args.output, proxy_assets=assets, progress=lambda message: print(message, flush=True))
        print(f'Benchmark complete: {args.output / "report.md"}')
    except KeyboardInterrupt:
        print('Benchmark interrupted. Completed evaluations and run manifests were retained.', file=sys.stderr)
        return 130
    except (ValueError, OSError, KeyError, OptimizationError) as exc:
        print(f'Benchmark error: {exc}', file=sys.stderr)
        return 2
    return 0
