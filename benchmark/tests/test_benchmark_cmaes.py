"""Check CMA adaptation and the invariants needed for a fair benchmark."""
from __future__ import annotations

import json
from types import SimpleNamespace
from aemec_opt.engine import ObjectiveResult
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from aemec_benchmark.runner import BenchmarkSettings, run_benchmark
from aemec_benchmark.cli import build_parser, build_problem, main
from aemec_benchmark.cmaes import MultiObjectiveCMAES


class CMAESTests(unittest.TestCase):
    def test_shared_points_and_failures_are_not_invented_parents(self):
        alg = MultiObjectiveCMAES(7, [1.1, 1.1])
        initial = np.array([[0, 1, .5], [.2, .3, .4], [.7, .2, .1], [1, 0, .5]])
        for i, point in enumerate(initial):
            returned, _ = alg.ask(point)
            np.testing.assert_array_equal(returned, point)
            alg.tell(returned, None if i == 1 else [point[0], 1-point[0]])
        point, details = alg.ask()
        self.assertEqual(details['parent_count'], 3)
        np.testing.assert_array_equal(alg.strategy.parents, initial[[0, 2, 3]])
        sigmas = np.asarray(alg.strategy.sigmas).copy()
        covariance_factors = np.asarray(alg.strategy.A).copy()
        alg.tell(point, None)
        np.testing.assert_array_equal(alg.strategy.sigmas, sigmas)
        np.testing.assert_array_equal(alg.strategy.A, covariance_factors)
        self.assertEqual(alg.updates, 0)
        self.assertEqual(alg.attempts, 5)

    def test_success_updates_covariance_and_population_with_both_objectives(self):
        alg = MultiObjectiveCMAES(7, [1.1, 1.1])
        point, _ = alg.ask([.3, .3, .3])
        alg.tell(point, [1., 1.])
        point, details = alg.ask()
        self.assertEqual(details['proposal'], 'mo_cmaes')
        alg.tell(point, [.5, .5])  # This offspring strictly dominates its parent.
        self.assertEqual(alg.updates, 1)
        np.testing.assert_array_equal(alg.strategy.parents[0], point)
        self.assertEqual(alg.strategy.parents[0].fitness.values, (.5, .5))
        self.assertFalse(np.allclose(alg.strategy.A[0], np.eye(3)))
        self.assertNotEqual(alg.strategy.sigmas[0], .2)

    def test_hypervolume_truncation_uses_fixed_reference(self):
        alg = MultiObjectiveCMAES(0, [1., 1.])
        front = [alg.individual_type([.5]*3) for _ in range(3)]
        for individual, values in zip(front, [(0., .9), (.5, .5), (.9, 0.)]):
            individual.fitness.values = values
        # Endpoint contributions are .05; the middle contributes .16. Removing
        # the first endpoint preserves the most HV; an adaptive ref=[2,2] would
        # instead remove the middle. Verify DEAP's passed ref cannot change it.
        self.assertEqual(alg._least_contributor(front, ref=[2., 2.]), 0)
        self.assertEqual(alg._least_contributor(front, ref=[100., 100.]), 0)

    def test_reflection_is_the_design_reported_and_used_for_adaptation(self):
        alg = MultiObjectiveCMAES(42, [1.1, 1.1])
        initial, _ = alg.ask([.5, .5, .5])
        alg.tell(initial, [1., 1.])
        point, _ = alg.ask()
        alg.tell(point, None)  # Construct the strategy without changing parents.
        offspring = alg.individual_type([-.2, 1.2, 3.4])
        offspring._ps = ('o', 0)
        with patch.object(alg.strategy, 'generate', return_value=[offspring]):
            point, details = alg.ask()
        np.testing.assert_allclose(point, [.2, .8, .6])
        self.assertEqual(details['reflected_coordinates'], 3)
        self.assertEqual(alg.pending._ps, ('o', 0))
        alg.tell(point, [.4, .4])
        np.testing.assert_array_equal(alg.strategy.parents[0], point)

    def test_reproducible_with_interleaved_runs_and_external_rng_unchanged(self):
        a, b = [MultiObjectiveCMAES(42, [1.1, 1.1]) for _ in range(2)]
        for alg in (a, b):
            for point in np.random.default_rng(5).uniform(size=(4, 3)):
                returned, _ = alg.ask(point)
                alg.tell(returned, [point[0], 1-point[0]+point[1]])
        external_state = np.random.get_state()
        try:
            for _ in range(25):
                state = np.random.get_state()
                x, _ = a.ask()
                after = np.random.get_state()
                np.testing.assert_array_equal(state[1], after[1])
                self.assertEqual(state[2:], after[2:])
                np.random.random(17)  # Unrelated global draws must not affect b.
                y, _ = b.ask()
                np.testing.assert_array_equal(x, y)
                self.assertTrue(np.all((x >= 0) & (x <= 1)))
                objectives = [x[0]+x[1]**2, 1-x[0]+x[2]**2]
                a.tell(x, objectives)
                b.tell(y, objectives)
        finally:
            np.random.set_state(external_state)

    def test_all_failed_initialization_recovers_without_free_evaluations(self):
        alg = MultiObjectiveCMAES(42, [1.1, 1.1])
        for _ in range(4):
            point, _ = alg.ask([.5, .5, .5])
            alg.tell(point, None)
        point, details = alg.ask()
        self.assertEqual(details['proposal'], 'sobol_fallback_no_successful_observations')
        alg.tell(point, None)
        point, _ = alg.ask()
        alg.tell(point, [.7, .8])
        point, details = alg.ask()
        self.assertEqual(details['proposal'], 'mo_cmaes')
        self.assertEqual(details['parent_count'], 1)
        alg.tell(point, [.5, .6])
        self.assertEqual(alg.attempts, 7)

    def test_pending_protocol_and_response_validation(self):
        alg = MultiObjectiveCMAES(42, [1.1, 1.1])
        with self.assertRaises(ValueError):
            alg.tell([.5]*3, [0., 0.])
        point, _ = alg.ask([.5]*3)
        with self.assertRaises(ValueError):
            alg.ask()
        for wrong_x, wrong_y in [([0]*3, [0, 0]), (point, [np.nan, 0]), (point, [1])]:
            with self.assertRaises(ValueError):
                alg.tell(wrong_x, wrong_y)
        alg.tell(point, None)

    def test_cli_cma_only_odd_budget_and_custom_sigma(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'run'
            with patch('aemec_benchmark.evaluation.shutil.which', return_value='/mock/tool'), \
                    patch('aemec_opt.case.AemecOpenFoamEvaluator') as adapter:
                adapter.return_value.evaluate.return_value = SimpleNamespace(
                    as_objective_result=lambda: ObjectiveResult((2., 3e-9), {}))
                self.assertEqual(main(['--algorithms', 'cmaes', '--output', str(out),
                                       '--initial', '4', '--budget', '9', '--seeds', '7', '--cma-sigma', '.35']), 0)
                self.assertEqual(adapter.return_value.evaluate.call_count, 9)
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['actual_evaluator_calls'], 9)
            self.assertEqual(manifest['settings']['cma_sigma'], .35)
            self.assertEqual(manifest['settings']['algorithms'], ['cmaes'])
            history = pd.read_csv(out/'seed_7/cmaes/history.csv')
            self.assertEqual(len(history), 9)
            self.assertTrue((history.failed == 0).all())
            self.assertTrue(np.all(np.diff(history.hypervolume) >= -1e-12))
            self.assertIn('MO-CMA-ES', (out/'report.md').read_text())

    def test_missing_dependency_fails_before_any_evaluation_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'run'
            args = build_parser().parse_args(['--output', str(out), '--dry-run'])
            problem = build_problem(args)
            with patch('aemec_benchmark.runner.require_deap', side_effect=ValueError('install DEAP')):
                with self.assertRaisesRegex(ValueError, 'install DEAP'):
                    run_benchmark(problem, BenchmarkSettings(algorithms=('cmaes',)), out)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
