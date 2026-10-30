"""Validate the measurement protocol without claiming CFD convergence."""
from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from aemec_opt.case import (
    AemecEvaluationConfig, AemecOpenFoamEvaluator, case_fingerprint, copy_clean_case,
    _named_block_span,
)
from aemec_opt.engine import OptimizationError
from aemec_opt.fixed_current import (
    FixedCurrentRun, configure_fixed_current, fixed_current_objective,
)


ROOT = Path(__file__).resolve().parents[2]


def preflight(case):
    return subprocess.run([sys.executable, str(case / 'check_case.py'), '--static'],
                          cwd=case, capture_output=True, text=True)


def solver_log(end=20., voltage=None, crossover=None, current=None):
    """Paired output every 0.1 clock units, with optional controlled drift."""
    lines = []
    for step in range(1, round(end * 10) + 1):
        t = step / 10
        v = 1.9 if voltage is None else voltage(t)
        cross = 2e-8 if crossover is None else crossover(t)
        j = -10000. if current is None else current(t)
        lines.extend((f'Time = {t:g}',
                      f'Controlled boundary current: signed = {j * 8e-5:g}, current density = {j:g} A/m2, voltage = {v:.12g}',
                      f'Hydrogen crossover objective: anode release = {cross:.12g} mol/s'))
    return '\n'.join(lines + ['End', ''])


class FixedCurrentTests(unittest.TestCase):
    def setUp(self):
        self.config = AemecEvaluationConfig()
        self.run = FixedCurrentRun(20., .1, 200)

    def test_default_uses_final_window_means_not_last_sample(self):
        log = solver_log(voltage=lambda t: 1.9 + t * .0001,
                         crossover=lambda t: 2e-8 * (1 + t * .0001))
        result = fixed_current_objective(log, self.config, self.run)
        self.assertEqual(result.window_sample_count, 51)
        self.assertEqual((result.window_start_s, result.window_end_s), (15, 20))
        self.assertAlmostEqual(result.cell_voltage_v, 1.90175)
        self.assertAlmostEqual(result.crossover_rate_mol_s, 2.0035e-8, places=18)
        self.assertNotEqual(result.cell_voltage_v, 1.902)

    def test_shorter_run_uses_its_own_final_window(self):
        config = replace(self.config, simulation_duration_s=10.)
        result = fixed_current_objective(solver_log(10), config, FixedCurrentRun(10, .1, 100))
        self.assertEqual((result.window_start_s, result.window_end_s), (5, 10))
        self.assertEqual(result.window_sample_count, 51)

    def test_stable_current_does_not_hide_crossover_or_voltage_drift(self):
        for arguments, message in (
            ({'crossover': lambda t: 1e-8 * (1 + .02 * t)}, 'crossover relative range'),
            ({'voltage': lambda t: 1.9 + .002 * t}, 'voltage range'),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(OptimizationError, message):
                fixed_current_objective(solver_log(**arguments), self.config, self.run)

    def test_every_sample_must_be_on_target_in_the_electrolysis_direction(self):
        for current in (lambda t: -9500, lambda t: 10000,
                        lambda t: -10200 if t == 17 else -10000):
            with self.subTest(current=current), self.assertRaisesRegex(OptimizationError, 'target-current error'):
                fixed_current_objective(solver_log(current=current), self.config, self.run)

    def test_current_range_check_remains_separate_from_target_tolerance(self):
        config = replace(self.config, current_relative_tolerance=.001)
        with self.assertRaisesRegex(OptimizationError, 'current relative range'):
            fixed_current_objective(solver_log(current=lambda t: -10000 + 5 * (t - 17.5)), config, self.run)

    def test_missing_nonfinite_duplicate_or_unfinished_data_are_rejected(self):
        valid = solver_log()
        block = ('Time = 17\nControlled boundary current: signed = -0.8, current density = -10000 A/m2, voltage = 1.9\n'
                 'Hydrogen crossover objective: anode release = 2e-08 mol/s\n')
        self.assertIn(block, valid)
        for invalid in (solver_log(19.9), valid.replace('End', ''), valid + 'FOAM FATAL ERROR',
                        valid.replace(block, ''), valid.replace(block, block + block),
                        valid.replace(block, block.replace('voltage = 1.9', 'voltage = nan')),
                        solver_log(crossover=lambda t: -1e-8), solver_log(voltage=lambda t: -1)):
            with self.subTest(log=invalid[-150:]), self.assertRaises(OptimizationError):
                fixed_current_objective(invalid, self.config, self.run)

    def test_zero_crossover_is_finite_and_accepted(self):
        result = fixed_current_objective(solver_log(crossover=lambda t: 0), self.config, self.run)
        self.assertEqual(result.crossover_relative_range, 0)

    def test_configuration_sets_current_and_both_control_files(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory) / 'case'
            copy_clean_case(ROOT / 'run/AEMEC', case)
            for duration in (20., 10.):
                config = replace(self.config, simulation_duration_s=duration, target_current_density_a_m2=8000)
                run = configure_fixed_current(case, config)
                self.assertEqual(run.outer_iterations, round(duration * 10))
                region = (case / 'constant/phiEAnode/regionProperties').read_text()
                self.assertRegex(region, r'galvanostatic\s*\{\s*active\s+true;')
                self.assertIn(f'values ((0 -8000) ({duration:g} -8000));', region)
                self.assertRegex(region, r'targets\s+\(-8000\);')
                self.assertRegex(region, rf'minimumHoldDuration\s+{duration:g};')
                for name in ('controlDict', 'controlDict.run'):
                    control = (case / 'system' / name).read_text()
                    self.assertRegex(control, rf'endTime\s+{duration:g};')
                    self.assertRegex(control, rf'writeInterval\s+{duration:g};')
                    self.assertRegex(control, r'deltaT\s+0.1;')
                checked = preflight(case)
                self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

    def test_preflight_rejects_mismatched_hold_and_accepts_supported_conductivity_models(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory) / 'case'
            copy_clean_case(ROOT / 'run/AEMEC', case)
            configure_fixed_current(case, replace(self.config, simulation_duration_s=10))
            region_path = case / 'constant/phiEAnode/regionProperties'
            original = region_path.read_text()
            region_path.write_text(original.replace('(10 -10000)', '(20 -10000)'))
            self.assertNotEqual(preflight(case).returncode, 0)
            region_path.write_text(original)

            membrane_path = case / 'constant/phiAnion/regionProperties'
            text = membrane_path.read_text()
            start, end = _named_block_span(text, 'membrane')
            for value in ('0', '-1', 'nan'):
                membrane_path.write_text(text[:start] + (
                    '\n sigmaModel constantSigma;\n constantSigmaCoeffs\n {\n'
                    f' cellZone membrane;\n sigma {value};\n }}\n') + text[end:])
                checked = preflight(case)
                self.assertNotEqual(checked.returncode, 0)
                self.assertIn('conductivity', checked.stderr)
            membrane_path.write_text(text[:start] + (
                '\n sigmaModel arrheniusSigma;\n arrheniusSigmaCoeffs\n {\n'
                ' cellZone membrane;\n T T;\n sigmaRef 5.998635;\n TRef 298.15;\n Ea 14900;\n }\n'
            ) + text[end:])
            checked = preflight(case)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

    def test_bad_duration_window_and_time_step_fail_before_running(self):
        for kwargs in ({'simulation_duration_s': 0}, {'simulation_duration_s': float('nan')},
                       {'simulation_duration_s': 5}, {'final_window_s': 0},
                       {'final_window_s': float('inf')}, {'target_current_relative_tolerance': 0},
                       {'evaluation_mode': 'invalid'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(OptimizationError):
                replace(self.config, **kwargs).validate()
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory) / 'case'
            copy_clean_case(ROOT / 'run/AEMEC', case)
            for kwargs in ({'simulation_duration_s': 10.05}, {'final_window_s': .1}, {'final_window_s': 5.05}):
                with self.subTest(kwargs=kwargs), self.assertRaises(OptimizationError):
                    configure_fixed_current(case, replace(self.config, **kwargs))

    def test_protocol_settings_distinguish_resumes(self):
        original = self.config.resume_settings()
        for kwargs in ({'simulation_duration_s': 10}, {'final_window_s': 3},
                       {'evaluation_mode': 'voltage-sweep'}, {'target_current_relative_tolerance': .005}):
            self.assertNotEqual(original, replace(self.config, **kwargs).resume_settings())

    def test_adapter_runs_once_preserves_source_and_keeps_failed_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            copy_clean_case(ROOT / 'run/AEMEC', source)
            before = case_fingerprint(source)
            output = solver_log()
            calls = []

            def command(argv, work, log, timeout):
                calls.append(argv[0])
                log.parent.mkdir(parents=True, exist_ok=True)
                if argv[0] == 'mesh':
                    shutil.copytree(work / '0.orig', work / '0')
                    for region in ('', *AemecOpenFoamEvaluator.required_mesh_regions):
                        path = work / 'constant' / region / 'polyMesh/points'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text('mock mesh')
                    # Mimic a wrapper recreating the active dictionary.
                    shutil.copyfile(work / 'system/controlDict.mesh', work / 'system/controlDict')
                    log.write_text('mesh complete')
                    return ''
                self.assertRegex((work / 'system/controlDict').read_text(), r'endTime\s+20;')
                self.assertRegex((work / 'system/controlDict').read_text(), r'deltaT\s+0.1;')
                log.write_text(output)
                return output

            evaluator = AemecOpenFoamEvaluator(source, root / 'work', root / 'logs',
                                             ('mesh',), ('solver',), self.config, None)
            with patch('aemec_opt.case.run_command', side_effect=command):
                result = evaluator.evaluate(0, 40, 333.15, .7)
                self.assertEqual(calls, ['mesh', 'solver'])
                metadata = result.as_objective_result().metadata
                self.assertEqual(metadata['evaluation_mode'], 'fixed-current')
                self.assertEqual(metadata['solver_outer_iterations'], 200)
                self.assertNotIn('interpolation_fraction', metadata)
                self.assertNotIn('sweep_end_s', metadata)
                self.assertGreaterEqual(metadata['duration_s'], 0)
                with result.timeseries_csv.open() as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 200)
                self.assertEqual(result.timeseries_plot.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
                output = solver_log(crossover=lambda t: 1e-8 * (1 + .02 * t))
                with self.assertRaisesRegex(OptimizationError, 'crossover relative range'):
                    evaluator.evaluate(1, 60, 343.15, .6)
                self.assertTrue((root / 'logs/trial_0001_timeseries.csv').is_file())
                self.assertTrue((root / 'logs/trial_0001_timeseries.png').is_file())
            self.assertEqual(case_fingerprint(source), before)


if __name__ == '__main__':
    unittest.main()
