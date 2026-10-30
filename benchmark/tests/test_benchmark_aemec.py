"""Check real-model routing without requiring OpenFOAM on the test machine.

The filesystem/configuration and optimization loop are real. Only executable
lookup and the expensive CFD adapter are mocked; this is not CFD validation.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from aemec_benchmark.cli import ROOT, build_parser, build_problem, main
from aemec_opt.engine import ObjectiveResult


class AemecBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root/'run/AEMEC'
        self.source.mkdir(parents=True)
        # Simulate a new checkout on another machine, with the bundled config
        # but deliberately no ELA/results, proxy files, or evaluated CSVs.
        self.config = self.root/'benchmark/configs/aemec_benchmark.json'
        self.config.parent.mkdir(parents=True)
        shutil.copyfile(ROOT/'benchmark/configs/aemec_benchmark.json', self.config)
        root_patch = patch('aemec_benchmark.cli.ROOT', self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.out = self.root/'output'

    def args(self, *options):
        return build_parser().parse_args(['--output', str(self.out), *map(str, options)])

    def test_default_dry_run_needs_no_ela_results_or_cfd_and_creates_no_output(self):
        stdout = io.StringIO()
        with patch('aemec_opt.case.AemecOpenFoamEvaluator') as adapter, \
                patch('aemec_benchmark.evaluation.shutil.which', side_effect=AssertionError('dry run checked executables')), \
                patch('aemec_benchmark.cli.resolve_proxy_paths', side_effect=AssertionError('used proxies')), \
                redirect_stdout(stdout):
            self.assertEqual(main(['--output', str(self.out), '--dry-run']), 0)
            adapter.assert_not_called()
        self.assertEqual(self.args().backend, 'aemec')
        self.assertFalse((self.root/'ELA').exists())
        self.assertFalse(self.out.exists())
        self.assertIn('1040 actual evaluator calls', stdout.getvalue())
        self.assertIn('Actual OpenFOAM fixed-current runs (20 clock units, final 5 averaged)', stdout.getvalue())
        self.assertIn('runtime compatibility were not checked', stdout.getvalue())

    def test_custom_configuration_and_objective_overrides_are_recorded(self):
        data = json.loads(self.config.read_text())
        data['input_upper'] = [80, 340, .7]
        data['objective_limits'] = [[1.5, 1e-9], [2.5, 1e-8]]
        custom = self.root/'custom.json'
        custom.write_text(json.dumps(data))
        args = self.args('--problem-config', custom, '--dry-run',
                         '--objective-lower', '1.6', '2e-9', '--objective-upper', '2.4', '8e-9',
                         '--reference', '2.5', '9e-9')
        problem = build_problem(args)
        np.testing.assert_array_equal(problem.upper, [80, 340, .7])
        np.testing.assert_array_equal(problem.objective_lower, [1.6, 2e-9])
        np.testing.assert_array_equal(problem.objective_upper, [2.4, 8e-9])
        np.testing.assert_array_equal(problem.reference, [2.5, 9e-9])
        self.assertEqual(problem.metadata['configuration']['path'], str(custom))
        self.assertEqual(problem.metadata['configuration']['kind'], 'problem_config')
        self.assertEqual(len(problem.metadata['configuration']['sha256']), 64)

    def test_fixed_current_duration_reaches_every_adapter_and_manifest(self):
        problem = build_problem(self.args('--dry-run', '--simulation-duration-s', '10', '--final-window-s', '3'))
        self.assertEqual(problem.metadata['evaluation']['simulation_duration_s'], 10)
        self.assertEqual(problem.metadata['evaluation']['final_window_s'], 3)
        self.assertEqual(problem.metadata['evaluation']['evaluation_mode'], 'fixed-current')
        with patch('aemec_opt.case.AemecOpenFoamEvaluator') as adapter:
            problem = build_problem(self.args('--dry-run', '--simulation-duration-s', '10', '--final-window-s', '3'))
            problem.evaluator_factory(self.out / 'seed_42/gp')
            config = adapter.call_args.kwargs['config']
            self.assertEqual((config.simulation_duration_s, config.final_window_s), (10, 3))
        sweep = build_problem(self.args('--dry-run', '--evaluation-mode', 'voltage-sweep'))
        self.assertEqual(sweep.metadata['evaluation']['evaluation_mode'], 'voltage-sweep')
        self.assertIsNone(sweep.metadata['evaluation']['simulation_duration_s'])

    def test_bad_duration_is_a_cli_error_without_cfd(self):
        with redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(main(['--output', str(self.out), '--dry-run', '--simulation-duration-s', '0']), 2)
        self.assertIn('Simulation duration', stderr.getvalue())

    def test_missing_openfoam_commands_fail_before_output_without_proxy_fallback(self):
        stderr = io.StringIO()
        with patch('aemec_benchmark.evaluation.shutil.which', side_effect=lambda name: '/usr/bin/make' if name == 'make' else None), \
                patch('aemec_benchmark.cli.resolve_proxy_paths', side_effect=AssertionError('used proxies')), \
                patch('aemec_opt.case.AemecOpenFoamEvaluator') as adapter, redirect_stderr(stderr):
            self.assertEqual(main(['--output', str(self.out)]), 2)
            adapter.assert_not_called()
        for name in ['openFuelCell', 'blockMesh', 'renumberMesh', 'topoSet', 'splitMeshRegions']:
            self.assertIn(name, stderr.getvalue())
        self.assertIn('No proxy fallback', stderr.getvalue())
        self.assertFalse(self.out.exists())

    def test_custom_command_wrappers_do_not_require_local_openfoam_commands(self):
        # A wrapper may supply its own environment (e.g. a container launcher).
        # Its internal tools cannot be checked on the host by shutil.which.
        with patch('aemec_benchmark.evaluation.shutil.which', side_effect=lambda name: '/bin/'+name) as lookup:
            build_problem(self.args('--mesh-command', 'mesh-wrapper --mesh',
                                    '--solver-command', 'solver-wrapper --solve'))
        self.assertEqual([call.args[0] for call in lookup.call_args_list], ['mesh-wrapper', 'solver-wrapper'])

    def test_malformed_bounds_and_limits_fail_without_evaluation(self):
        original = json.loads(self.config.read_text())
        for key, value, error in [
            ('input_upper', [100, 353.15], 'upper must contain 3'),
            ('input_lower', [10, 298.15, 0], 'porosity strictly inside'),
            ('objective_limits', [1, 2], 'objective_limits must contain'),
            ('objective_limits', [[2, 1e-9], [1, 2e-9]], 'Upper bounds/scales'),
            ('objective_columns', ['crossover_rate_mol_s', 'cell_voltage_v'], 'incompatible'),
        ]:
            with self.subTest(key=key, value=value):
                data = dict(original, **{key: value})
                self.config.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, error):
                    build_problem(self.args('--dry-run'))

    def test_proxy_source_flags_cannot_silently_select_synthetic_evaluation(self):
        with redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(main(['--output', str(self.out), '--proxy-root', str(self.root/'some_proxies'), '--dry-run']), 2)
        self.assertIn('require --algorithms ela_proxy', stderr.getvalue())

    def test_all_algorithms_use_cfd_adapter_and_store_physical_observations(self):
        # Four real initial adapter calls, then one adaptive call per algorithm.
        # The mocked values make this test fast; no surrogate/proxy callback is
        # substituted by the benchmark itself, and no extra observations appear.
        observations = []
        def evaluate(number, thickness, temperature, porosity):
            result = ObjectiveResult((1.96 + .001*thickness + .00001*(temperature-298.15),
                                      4.3e-9 - 2e-11*thickness + 1e-10*porosity), {})
            observations.append(result.values)
            return SimpleNamespace(as_objective_result=lambda: result)

        with patch('aemec_opt.case.AemecOpenFoamEvaluator') as adapter, \
                patch('aemec_benchmark.evaluation.shutil.which', return_value='/mock/executable'), \
                patch('aemec_benchmark.cli.resolve_proxy_paths', side_effect=AssertionError('used proxies')), \
                redirect_stdout(io.StringIO()):
            adapter.return_value.evaluate.side_effect = evaluate
            self.assertEqual(main(['--output', str(self.out), '--budget', '5', '--initial', '4',
                                   '--seeds', '42', '--gp-candidates', '16', '--gp-restarts', '0']), 0)
            self.assertEqual(adapter.return_value.evaluate.call_count, 8)
            self.assertEqual(adapter.call_count, 5)  # Shared pool and four algorithms.
            work_cases = [call.kwargs['work_case'] for call in adapter.call_args_list]
            self.assertEqual(len(set(work_cases)), 5)
            self.assertNotIn(self.source, work_cases)

        names = ['cell_voltage_v', 'crossover_rate_mol_s']
        for index, algorithm in enumerate(['gp', 'tpe', 'nsga2', 'cmaes']):
            rows = pd.read_csv(self.out/f'seed_42/{algorithm}/evaluations.csv')
            self.assertEqual(len(rows), 5)
            np.testing.assert_allclose(rows[names].iloc[:4], observations[:4])
            np.testing.assert_allclose(rows[names].iloc[4], observations[4+index])
        manifest = json.loads((self.out/'manifest.json').read_text())
        self.assertEqual(manifest['actual_evaluator_calls'], 8)
        self.assertIn('actual AEMEC/OpenFOAM CFD runs', (self.out/'report.md').read_text())
        self.assertIn('Evaluation protocol: fixed-current', (self.out/'report.md').read_text())
        self.assertFalse((self.root/'ELA').exists())


if __name__ == '__main__':
    unittest.main()
