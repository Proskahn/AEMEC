"""The production benchmark evaluator: actual AEMEC/OpenFOAM CFD runs.

Algorithm models and proxy scores never enter this module or supply responses.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import numpy as np

from .runner import INPUTS, BenchmarkProblem


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_cfd_problem(args, ROOT):
    """Configure a shared CFD contract; constructing it makes no solver calls."""
    if getattr(args, 'backend', 'aemec') != 'aemec':
        raise ValueError('CFD is the only benchmark evaluator; proxies belong to the ela_proxy algorithm')
    metadata = {'backend': 'aemec', 'evaluator': 'cfd'}
    from aemec_opt.case import AemecEvaluationConfig, AemecOpenFoamEvaluator, case_fingerprint, validate_path_layout
    from aemec_opt.protocol import protocol_options
    path = (args.analysis.expanduser()/'metadata.json' if args.analysis is not None else
            (args.problem_config or ROOT/'benchmark/configs/aemec_benchmark.json').expanduser())
    if not path.is_file():
        raise ValueError(f'AEMEC bounds/scales file not found: {path}. '
                         'Use the bundled benchmark/configs/aemec_benchmark.json with no --analysis option, '
                         'or supply --problem-config pointing to an existing configuration.')
    study = json.loads(path.read_text())
    # Read metadata only. The 128/256 previously evaluated response rows are
    # not handed to any optimizer as free extra initialization observations.
    if not isinstance(study, dict) or study.get('input_columns') != INPUTS or study.get('objective_columns') != ['cell_voltage_v', 'crossover_rate_mol_s']:
        raise ValueError('AEMEC bounds/scales file has incompatible inputs/objectives')
    lower, upper = np.asarray(study['input_lower'], dtype=float), np.asarray(study['input_upper'], dtype=float)
    limits = np.asarray(study['objective_limits'], dtype=float)
    if limits.shape != (2, 2):
        raise ValueError('AEMEC objective_limits must contain [lower, upper], each with voltage and crossover endpoints')
    objective_lower, objective_upper = limits
    names = ('cell_voltage_v', 'crossover_rate_mol_s')
    # Every algorithm shares the same protocol and endpoint checks. The default
    # is one fixed-current run; a complete sweep is an explicit comparison mode.
    config = AemecEvaluationConfig(
        **protocol_options(args),
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
        executables = [args.mesh_command[0], args.solver_command[0]]
        if args.mesh_command == ('make', 'mesh'):
            # Checking only `make` misses the common case where OpenFOAM
            # has not been sourced. These commands are invoked internally
            # by the bundled Makefile/preprocessing/check_case scripts.
            executables.extend(['blockMesh', 'renumberMesh', 'topoSet', 'splitMeshRegions', 'python3'])
        missing = [name for name in dict.fromkeys(executables) if shutil.which(name) is None]
        if missing:
            raise ValueError('Required CFD executable(s) not found: '+', '.join(missing)+'. '
                             'Load the MPI module, source your OpenFOAM etc/bashrc, and build the '
                             'AEMEC solver/tools with src/Allwmake. No proxy fallback is used.')
    metadata.update({'source_case': str(args.case.resolve()), 'case_sha256': case_fingerprint(args.case),
                     'evaluation': config.resume_settings(), 'mesh_command': list(args.mesh_command),
                     'solver_command': list(args.solver_command), 'work_root': str(work_root.resolve()),
                     'configuration': {'kind': 'analysis_metadata' if args.analysis is not None else 'problem_config',
                                       'path': str(path.resolve()), 'sha256': file_hash(path)},
                     'note': 'Configuration supplies fixed bounds/scales only. Every observation comes from '
                             f'a new OpenFOAM {config.evaluation_mode} run; existing response rows do not train any optimizer.'})
    if args.analysis is not None:
        # Preserve these manifest fields for callers that previously used
        # --analysis, while documenting either configuration source above.
        metadata.update({'analysis_metadata': str(path.resolve()), 'analysis_sha256': file_hash(path)})
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
    # Apply explicit study-wide overrides to every algorithm. Requiring
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
    if (lower[0] <= 0 or lower[1] <= 0 or not 0 < lower[2] < upper[2] < 1):
        raise ValueError('AEMEC bounds require positive thickness/temperature and porosity strictly inside (0, 1)')
    return problem

