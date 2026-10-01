"""Sequential, checkpointed evaluation of a fixed AEMEC design (no optimization)."""
from __future__ import annotations

import csv
from contextlib import contextmanager, ExitStack
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ('membrane_thickness_um', 'water_inlet_temperature_k', 'ptl_porosity')
OUTPUTS = ('cell_voltage_v', 'crossover_rate_mol_s')
COLUMNS = ('sample_id', *INPUTS, *OUTPUTS, 'status', 'failure_reason', 'attempts',
           'ptl_side', 'target_current_density_a_m2', 'duration_s', 'mesh_log',
           'solver_log', 'polarization_curve_csv', 'polarization_curve_plot', 'metadata_json')


def command(text):
    parts = tuple(shlex.split(text))
    if not parts:
        raise ValueError('External commands cannot be empty')
    return parts


def add_parser(subparsers):
    parser = subparsers.add_parser('evaluate', help='Run the fixed design through OpenFOAM, with resume')
    parser.add_argument('csv', type=Path, help='CSV written by the sample command')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'ELA/results/evaluation')
    parser.add_argument('--case', type=Path, default=ROOT/'run/AEMEC')
    parser.add_argument('--work-dir', type=Path, help='Dedicated scratch directory; defaults to ELA/work/<batch hash>')
    parser.add_argument('--mesh-command', type=command, default=('make', 'mesh'))
    parser.add_argument('--solver-command', type=command, default=('openFuelCell',))
    parser.add_argument('--ptl-side', choices=['anode', 'cathode', 'both'], default='anode')
    parser.add_argument('--target-current-density-a-m2', type=float, default=10000.)
    parser.add_argument('--current-relative-tolerance', type=float)
    parser.add_argument('--crossover-stability-relative-tolerance', type=float)
    parser.add_argument('--voltage-stability-tolerance-v', type=float)
    parser.add_argument('--stability-samples', type=int)
    parser.add_argument('--timeout-minutes', type=float, help='Wall-clock timeout for each mesh/solver command')
    parser.add_argument('--limit', type=int, help='Maximum attempts in this invocation (e.g. 1 for a pilot)')
    parser.add_argument('--max-consecutive-failures', type=int, default=3)
    parser.add_argument('--retry-failed', action='store_true', help='Retry failed rows; completed rows remain skipped')
    parser.add_argument('--dry-run', action='store_true', help='Validate design/settings and show pending count without running CFD')


def load_design(path):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'sample_id', *INPUTS} <= set(reader.fieldnames or []):
            raise ValueError(f'Design CSV requires sample_id and {", ".join(INPUTS)}')
        rows, ids, designs = [], set(), set()
        for line, row in enumerate(reader, 2):
            try:
                sample_id = int(row['sample_id'])
                values = tuple(float(row[k]) for k in INPUTS)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'Invalid sample ID or numeric inputs on CSV line {line}') from exc
            if sample_id < 0 or sample_id in ids:
                raise ValueError(f'Duplicate or negative sample_id on CSV line {line}')
            if not all(math.isfinite(v) for v in values) or values[0] <= 0 or values[1] <= 0 or not 0 < values[2] < 1:
                raise ValueError(f'Invalid physical inputs on CSV line {line}')
            if values in designs:
                raise ValueError(f'Duplicate design on CSV line {line}')
            ids.add(sample_id)
            designs.add(values)
            rows.append({'sample_id': sample_id, **dict(zip(INPUTS, values)),
                         'status': 'pending', 'attempts': 0})
    if not rows:
        raise ValueError('Design CSV is empty')
    return rows


def atomic_text(path, text):
    """Replace one file atomically so interruption cannot leave a truncated checkpoint."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False, encoding='utf-8', newline='') as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def checkpoint(output, settings, rows):
    # JSON is authoritative. On resume the CSV is rebuilt if interrupted between replacements.
    atomic_text(output/'checkpoint.json', json.dumps({'settings': settings, 'rows': rows},
                                                    indent=2, allow_nan=False)+'\n')
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(output/'evaluated.csv', buffer.getvalue())


@contextmanager
def lock(path):
    """Linux/macOS advisory lock, released automatically even on process termination."""
    import fcntl
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'Another evaluation is using {path.parent}') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def check_executables(mesh_command, solver_command):
    executables = [mesh_command[0], solver_command[0]]
    if mesh_command == ('make', 'mesh'):
        executables += ['blockMesh', 'renumberMesh', 'topoSet', 'splitMeshRegions']
    missing = [name for name in executables if '/' not in name and shutil.which(name) is None]
    if missing:
        raise ValueError('Missing executables: '+', '.join(missing)+
                         '. Source your OpenFOAM environment and build the repository solver/tools first.')


def run_evaluation(args):
    # Keep sample/analyze usable without installing the optimizer's dependencies.
    sys.path.insert(0, str(ROOT/'opt'))
    from aemec_opt.case import (AemecEvaluationConfig, AemecOpenFoamEvaluator,
                               case_fingerprint, validate_path_layout)
    from aemec_opt.engine import OptimizationError

    design_path = args.csv.resolve()
    output = args.output_dir.resolve()
    batch_id = hashlib.sha256(str(output).encode()).hexdigest()[:12]
    work = (args.work_dir or ROOT/'ELA/work'/batch_id).resolve()
    source = args.case.resolve()
    scratch = work/'case'
    validate_path_layout(source, scratch, output)
    # The work root and output receive locks/markers; neither may be inside the source case.
    if work == source or source in work.parents or work in source.parents:
        raise ValueError('Work directory and source case must not overlap')
    if scratch == design_path or scratch in design_path.parents:
        raise ValueError('Design CSV cannot be inside the disposable scratch case')
    if design_path in (output/'checkpoint.json', output/'evaluated.csv', output/'.evaluation.lock'):
        raise ValueError('Design CSV must not be an output artifact')
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    if args.max_consecutive_failures < 1:
        raise ValueError('--max-consecutive-failures must be positive')
    if args.timeout_minutes is not None and (not math.isfinite(args.timeout_minutes) or args.timeout_minutes <= 0):
        raise ValueError('--timeout-minutes must be finite and positive')
    config_options = {key: getattr(args, key) for key in (
        'ptl_side', 'target_current_density_a_m2', 'current_relative_tolerance',
        'crossover_stability_relative_tolerance', 'voltage_stability_tolerance_v',
        'stability_samples') if getattr(args, key) is not None}
    config = AemecEvaluationConfig(**config_options)
    config.validate()
    rows = load_design(design_path)
    settings = {
        'schema': 1, 'design_sha256': hashlib.sha256(design_path.read_bytes()).hexdigest(),
        'source_case': str(source), 'case_fingerprint': case_fingerprint(source),
        'adapter_sha256': hashlib.sha256((ROOT/'opt/aemec_opt/case.py').read_bytes()).hexdigest(),
        'evaluation': config.resume_settings(), 'work_case': str(scratch),
        'mesh_command': list(args.mesh_command), 'solver_command': list(args.solver_command),
        'timeout_minutes': args.timeout_minutes,
    }

    def restore():
        path = output/'checkpoint.json'
        if not path.exists():
            if output.exists() and any(p.name != '.evaluation.lock' for p in output.iterdir()):
                raise ValueError('Output directory is not empty and has no checkpoint; choose a new --output-dir')
            return rows
        saved = json.loads(path.read_text())
        if saved['settings'] != settings:
            raise ValueError('Resume settings differ (design, case, adapter, commands, or evaluation settings). '
                             'Use a new --output-dir for the changed study.')
        restored = saved['rows']
        identity = lambda row: tuple(row[k] for k in ('sample_id', *INPUTS))
        if list(map(identity, restored)) != list(map(identity, rows)):
            raise ValueError('Checkpoint rows do not match the design CSV')
        return restored

    def eligible(row):
        return row['status'] in ('pending', 'running', 'interrupted') or (args.retry_failed and row['status'] == 'failed')

    if args.dry_run:
        restored = restore()
        count = sum(eligible(r) for r in restored)
        print(f'{len(rows)} designs; {count} eligible; up to {min(count, args.limit or count)} attempts this invocation.\n'
              f'Source: {source}\nScratch: {scratch}\nResults: {output}/evaluated.csv\n'
              'Dry run: no CFD calls or output files written; executable availability was not checked.', flush=True)
        return 0

    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for directory in sorted({output, work}):
            stack.enter_context(lock(directory/'.evaluation.lock'))
        rows = restore()
        owner_path = work/'.ela-owner.json'
        owner = {'output_dir': str(output)}
        if owner_path.exists() and json.loads(owner_path.read_text()) != owner:
            raise ValueError('Scratch directory belongs to another batch; choose a different --work-dir')
        if scratch.exists() and not owner_path.exists():
            raise ValueError('Scratch case already exists without an ELA ownership marker; choose an empty --work-dir')
        if any(eligible(r) for r in rows):
            check_executables(args.mesh_command, args.solver_command)
        atomic_text(owner_path, json.dumps(owner)+'\n')
        checkpoint(output, settings, rows)
        attempts, failures = 0, 0
        for row in rows:
            if not eligible(row):
                continue
            if args.limit is not None and attempts >= args.limit:
                break
            attempts += 1
            row['attempts'] += 1
            logs = output/'logs'/f"sample_{row['sample_id']:04d}"/f"attempt_{row['attempts']:03d}"
            trial = f"trial_{row['sample_id']:04d}"
            row.update({key: None for key in OUTPUTS})
            row.update(status='running', failure_reason='', duration_s=None, metadata_json='',
                       ptl_side=config.ptl_side, target_current_density_a_m2=config.target_current_density_a_m2,
                       mesh_log=str(logs/f'{trial}_mesh.log'), solver_log=str(logs/f'{trial}_solver.log'),
                       polarization_curve_csv=str(logs/f'{trial}_polarization_curve.csv'),
                       polarization_curve_plot=str(logs/f'{trial}_polarization_curve.png'))
            checkpoint(output, settings, rows)
            print(f"Sample {row['sample_id']} (attempt {row['attempts']}): "
                  +', '.join(f'{key}={row[key]:g}' for key in INPUTS), flush=True)
            evaluator = AemecOpenFoamEvaluator(
                source_case=source, work_case=scratch, logs_dir=logs,
                mesh_command=args.mesh_command, solver_command=args.solver_command, config=config,
                timeout_s=None if args.timeout_minutes is None else args.timeout_minutes*60.)
            started = time.monotonic()
            try:
                result = evaluator.evaluate(row['sample_id'], *(row[k] for k in INPUTS)).as_objective_result()
                result.validate(2)
                if result.values[0] <= 0 or result.values[1] < 0:
                    raise OptimizationError('Evaluator returned nonphysical objectives')
                row.update(zip(OUTPUTS, result.values))
                row.update(status='complete', metadata_json=json.dumps(dict(result.metadata), allow_nan=False))
                failures = 0
                print(f'  complete: V={result.values[0]:.8g} V, crossover={result.values[1]:.8g} mol/s', flush=True)
            except KeyboardInterrupt:
                row.update(status='interrupted', failure_reason='Interrupted; rerun the same command to resume')
                raise
            except (OptimizationError, OSError) as exc:
                row.update(status='failed', failure_reason=str(exc))
                failures += 1
                print(f'  failed: {exc}', flush=True)
            except Exception as exc:
                row.update(status='failed', failure_reason=f'Unexpected error: {exc}')
                raise
            finally:
                row['duration_s'] = time.monotonic()-started
                checkpoint(output, settings, rows)
            if failures >= args.max_consecutive_failures:
                print('Stopped after consecutive failures. Inspect logs before resuming.', flush=True)
                break
        counts = {status: sum(r['status'] == status for r in rows) for status in ('complete', 'failed', 'pending', 'interrupted', 'running')}
        print(f'Results: {output}/evaluated.csv\nStatus counts: {counts}', flush=True)
        return 1 if counts['failed'] else 0
