from __future__ import annotations

import json
import io
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import optuna

from aemec_benchmark.runner import BenchmarkProblem, BenchmarkSettings, run_benchmark
from aemec_benchmark.algorithms import GaussianProcessBO, OptunaAlgorithm
from aemec_benchmark.cli import build_parser, build_problem, main
from aemec_benchmark.metrics import expected_hypervolume_improvement, hypervolume_2d, pareto_indices
from aemec_opt.engine import ObjectiveResult, OptimizationError


def synthetic_problem(factory=None):
    def default_factory(folder):
        def evaluate(number, x):
            return ObjectiveResult((float(x[0] + .2*x[1]), float(1-x[0] + .2*x[2])), {})
        return evaluate
    return BenchmarkProblem(np.zeros(3), np.ones(3), np.zeros(2), np.ones(2),
                            np.array([1.5, 1.5]), ('f1', 'f2'), {'backend': 'unit_test'},
                            default_factory if factory is None else factory)


class MetricTests(unittest.TestCase):
    def test_hypervolume_dominance_ties_and_reference(self):
        points = [[1, 4], [3, 2], [3, 2], [4, 4], [6, 1]]
        self.assertEqual(pareto_indices(points).tolist(), [0, 1, 4])
        self.assertEqual(hypervolume_2d(points, [5, 5]), 8.)
        self.assertEqual(hypervolume_2d([], [5, 5]), 0.)
        self.assertEqual(hypervolume_2d([[5, 1], [1, 5]], [5, 5]), 0.)
        self.assertEqual(hypervolume_2d([[1, 4], [3, 2]], [5, 5]), 8.)

    def test_deterministic_ehvi_matches_actual_hypervolume_increments(self):
        front, ref = np.array([[.2, .8], [.6, .3]]), np.array([1.1, 1.1])
        points = np.array([[.1, .2], [.3, .9], [.5, .5], [1.2, .1], [-.2, 0.]])
        result = expected_hypervolume_improvement(points, np.zeros_like(points), front, ref)
        expected = [hypervolume_2d(np.vstack([front, p]), ref)-hypervolume_2d(front, ref) for p in points]
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_stochastic_ehvi_against_independent_monte_carlo(self):
        front, ref = np.array([[.2, .8], [.6, .3]]), np.array([1.1, 1.1])
        mean, std = np.array([[.4, .5]]), np.array([[.2, .25]])
        analytic = expected_hypervolume_improvement(mean, std, front, ref)[0]
        draws = np.random.default_rng(2).normal(mean[0], std[0], size=(100000, 2))
        # Integrate improvement for each draw via deterministic rectangle strips.
        # Compare a subset against direct HV unions to independently verify strips.
        base = hypervolume_2d(front, ref)
        direct = np.array([hypervolume_2d(np.vstack([front, p]), ref)-base for p in draws[:2000]])
        increments = expected_hypervolume_improvement(draws, np.zeros_like(draws), front, ref)
        np.testing.assert_allclose(direct, increments[:2000], atol=1e-14)
        self.assertLess(abs(analytic-increments.mean()), 5*increments.std()/np.sqrt(len(draws)))


class AlgorithmTests(unittest.TestCase):
    def test_nsga_initial_population_then_offspring_and_tpe_initial_identity(self):
        initial = np.random.default_rng(0).uniform(0, 1, (4, 3))
        for name in ('tpe', 'nsga2'):
            alg = OptunaAlgorithm(name, 4, 42)
            for point in initial:
                returned, _ = alg.ask(point)
                np.testing.assert_array_equal(returned, point)
                alg.tell(returned, [returned[0], 1-returned[0]])
            point, _ = alg.ask()
            self.assertTrue(np.all((point >= 0) & (point <= 1)))
            if name == 'nsga2':
                trial = alg.study.get_trials()[-1]
                generation = alg.study.sampler.get_trial_generation(alg.study, trial)
                self.assertEqual(generation, 1)
            alg.tell(point, None)
            self.assertEqual(alg.study.get_trials()[-1].state, optuna.trial.TrialState.FAIL)

    def test_gp_uses_only_successful_data_and_removes_fitted_noise(self):
        alg = GaussianProcessBO(42, [1.1, 1.1], candidates=16, restarts=0)
        for point in np.random.default_rng(1).uniform(size=(4, 3)):
            alg.tell(point, [point[0], 1-point[0]])
        alg.tell([.1, .2, .3], None)
        captured = []
        original = expected_hypervolume_improvement
        def acquisition(mean, std, observed, reference):
            captured.append((mean.copy(), std.copy(), np.asarray(observed).copy()))
            return original(mean, std, observed, reference)
        with patch('aemec_benchmark.algorithms.expected_hypervolume_improvement', side_effect=acquisition):
            point, details = alg.ask()
        self.assertEqual(len(captured[0][2]), 4)
        self.assertTrue(np.isfinite(captured[0][1]).all())
        self.assertTrue(np.all(captured[0][1] >= 0))
        self.assertEqual(details['proposal'], 'analytic_ehvi')
        self.assertFalse(any(np.allclose(point, previous) for previous in alg.attempts))


class RunnerTests(unittest.TestCase):
    def test_shared_initialization_budgets_failures_and_metrics(self):
        calls = []
        def factory(folder):
            def evaluate(number, x):
                calls.append((folder.name, number, x.copy()))
                if number in (0, 5):
                    raise OptimizationError('test failure')
                return ObjectiveResult((float(x[0]), float(1-x[0]+.1*x[1])), {})
            return evaluate
        settings = BenchmarkSettings(budget=8, initial=4, seeds=(42,), gp_candidates=16, gp_restarts=0)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'run'
            results = run_benchmark(synthetic_problem(factory), settings, out, progress=lambda _: None)
            self.assertEqual(len(calls), 4+4*4)
            self.assertEqual(len(results), 4)
            self.assertTrue((results.successful == 6).all())
            self.assertTrue((results.failed == 2).all())
            initial = []
            for name in settings.algorithms:
                f = pd.read_csv(out/f'seed_42/{name}/evaluations.csv')
                self.assertEqual(len(f), 8)
                initial.append(f.iloc[:4][['membrane_thickness_um', 'water_inlet_temperature_k', 'ptl_porosity']].to_numpy())
                h = pd.read_csv(out/f'seed_42/{name}/history.csv')
                self.assertTrue(np.all(np.diff(h.hypervolume) >= -1e-12))
                self.assertEqual(h.iloc[5].hypervolume, h.iloc[4].hypervolume)
            for x in initial[1:]:
                np.testing.assert_array_equal(x, initial[0])
            m = json.loads((out/'manifest.json').read_text())
            self.assertEqual(m['actual_evaluator_calls'], 20)
            self.assertNotIn('deap', m['versions'])
            self.assertEqual(m['versions']['optuna'], '4.9.0')
            self.assertEqual(m['cmaes_protocol']['population_size'], 4)
            self.assertIn('OptunaHub MoCmaSampler', m['cmaes_protocol']['implementation'])
            self.assertEqual(m['status'], 'complete')
            self.assertEqual(m['reference_normalized'], [1.5, 1.5])
            for name in ['algorithm_summary.csv', 'paired_hypervolume.csv', 'hypervolume.png', 'pareto_fronts.png', 'report.md']:
                self.assertTrue((out/name).is_file())
            with self.assertRaisesRegex(ValueError, 'already exists'):
                run_benchmark(synthetic_problem(), settings, out)

    def test_repeated_seed_reproducibility_and_zero_success_report(self):
        settings = BenchmarkSettings(budget=6, initial=4, seeds=(2,), algorithms=('tpe', 'nsga2', 'cmaes'))
        def factory(folder):
            def fail(number, x):
                raise OptimizationError('unavailable')
            return fail
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp)/'a', Path(temp)/'b'
            for out in (a, b):
                summary = run_benchmark(synthetic_problem(factory), settings, out, progress=lambda _: None)
                self.assertTrue((summary.final_hypervolume == 0).all())
            for name in settings.algorithms:
                first = pd.read_csv(a/f'seed_2/{name}/evaluations.csv')
                second = pd.read_csv(b/f'seed_2/{name}/evaluations.csv')
                cols = ['membrane_thickness_um', 'water_inlet_temperature_k', 'ptl_porosity', 'status']
                pd.testing.assert_frame_equal(first[cols], second[cols])

    def test_configuration_rejects_invalid_budgets_scales_and_seeds(self):
        for settings in [BenchmarkSettings(budget=4, initial=4), BenchmarkSettings(initial=3),
                         BenchmarkSettings(seeds=(42, 42)), BenchmarkSettings(gp_candidates=0),
                         BenchmarkSettings(proxy_candidates=0), BenchmarkSettings(gp_restarts=-1)]:
            with self.assertRaises(ValueError):
                settings.validate()
        problem = synthetic_problem()
        problem.objective_upper = problem.objective_lower.copy()
        with self.assertRaises(ValueError):
            problem.validate()


class AdapterTests(unittest.TestCase):
    def test_cli_rejects_synthetic_evaluators_and_cfd_dry_run_is_free(self):
        for backend in ('proxies', 'zdt1'):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                build_parser().parse_args(['--backend', backend, '--output', 'unused'])
            self.assertEqual(error.exception.code, 2)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'dry'
            self.assertEqual(main(['--output', str(out), '--dry-run']), 0)
            self.assertFalse(out.exists())

    def test_aemec_adapter_preserves_parameter_order_and_scratch_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, analysis = root/'case', root/'analysis'
            source.mkdir()
            analysis.mkdir()
            (analysis/'metadata.json').write_text(json.dumps({
                'input_columns': ['membrane_thickness_um', 'water_inlet_temperature_k', 'ptl_porosity'],
                'objective_columns': ['cell_voltage_v', 'crossover_rate_mol_s'],
                'input_lower': [10, 298.15, .4], 'input_upper': [100, 353.15, .8],
                'objective_limits': [[1.5, 1e-9], [2.5, 1e-8]]}))
            args = build_parser().parse_args(['--backend', 'aemec', '--case', str(source),
                                             '--analysis', str(analysis), '--output', str(root/'out'),
                                             '--work-dir', str(root/'work'), '--dry-run'])
            with patch('aemec_opt.case.AemecOpenFoamEvaluator') as adapter:
                adapter.return_value.evaluate.return_value.as_objective_result.return_value = ObjectiveResult((2., 5e-9), {})
                problem = build_problem(args)
                evaluate = problem.evaluator_factory(root/'out/seed_42/gp')
                value = evaluate(7, [50, 333, .6])
                adapter.return_value.evaluate.assert_called_once_with(7, 50., 333., .6)
                self.assertEqual(value.values, (2., 5e-9))
                kwargs = adapter.call_args.kwargs
                self.assertNotEqual(kwargs['source_case'], kwargs['work_case'])
                self.assertNotIn(root/'out', kwargs['work_case'].parents)


if __name__ == '__main__':
    unittest.main()
