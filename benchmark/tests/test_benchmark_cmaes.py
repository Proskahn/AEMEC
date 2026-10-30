"""Verify the OptunaHub integration and CFD accounting, not the CMA paper itself."""
from __future__ import annotations

from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import optuna
import pandas as pd

from aemec_opt.engine import ObjectiveResult
from aemec_benchmark.algorithms import OptunaAlgorithm
from aemec_benchmark.runner import BenchmarkSettings, run_benchmark
from aemec_benchmark.cli import build_parser, build_problem, main
from aemec_benchmark.cmaes import SOURCE_PATH, require_mocma

INITIAL = np.array([[0, 1, .5], [.2, .3, .4], [.7, .2, .1], [1, 0, .5]])


def initialize(alg, failed=(), values=(1., 1.)):
    """Replay the same measured initial observations used by the runner."""
    for i, point in enumerate(INITIAL):
        returned, _ = alg.ask(point)
        np.testing.assert_array_equal(returned, point)
        alg.tell(returned, None if i in failed else values)


def study_attrs(alg):
    return alg.study._storage.get_study_system_attrs(alg.study._study_id)


class CMAESTests(unittest.TestCase):
    def test_pinned_source_license_and_runtime(self):
        self.assertEqual(hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
                         'a0a6fb2a9e32d9025c20ebfb0a05693d3cbb5b66fee18f396ddf1135d6793cf4')
        self.assertIn('Copyright (c) 2024 Yoshihiko Ozaki',
                      SOURCE_PATH.with_name('LICENSE').read_text())
        self.assertEqual(require_mocma().__name__, 'MoCmaSampler')
        with patch('aemec_benchmark.cmaes.optuna.__version__', '5.0.0'):
            with self.assertRaisesRegex(ValueError, 'Optuna 4.9.0'):
                require_mocma()

    def test_fixed_initial_trials_are_tagged_once_and_seed_the_real_population(self):
        alg = OptunaAlgorithm('cmaes', 4, 7)
        sampler = alg.study.sampler
        self.assertIsInstance(sampler, require_mocma())
        with patch.object(sampler, 'sample_relative', wraps=sampler.sample_relative) as sample:
            initialize(alg)
            self.assertEqual(sample.call_count, 4)
        trials = alg.study.get_trials()
        self.assertEqual({t.system_attrs['mocma:g'] for t in trials}, {0})
        self.assertEqual({t.system_attrs['mocma:k'] for t in trials}, set(range(4)))
        point, details = alg.ask()
        self.assertEqual(details['generation'], 1)
        self.assertEqual(details['population_size'], 4)
        self.assertEqual(details['proposal'], 'optunahub_mocma')
        self.assertEqual(set(study_attrs(alg)['mocma:generation:0:elite_ids']), {0, 1, 2, 3})
        self.assertIn(details['parent_trial_id'], range(4))
        for trial in trials:
            np.testing.assert_array_equal(list(trial.params.values()), INITIAL[trial.number])
            self.assertEqual(trial.values, [1., 1.])
        alg.tell(point, [.5, .5])

    def test_shared_failures_are_not_invented_parents_and_delay_population(self):
        alg = OptunaAlgorithm('cmaes', 4, 7)
        initialize(alg, failed=(1,))
        self.assertEqual(alg.study.trials[1].state, optuna.trial.TrialState.FAIL)
        self.assertIsNone(alg.study.trials[1].values)
        point, details = alg.ask()
        self.assertEqual(details['generation'], 0)
        alg.tell(point, [.2, .3])
        point, details = alg.ask()
        self.assertEqual(details['generation'], 1)
        self.assertEqual(set(study_attrs(alg)['mocma:generation:0:elite_ids']), {0, 2, 3, 4})
        self.assertEqual(alg.study.trials[4].values, [.2, .3])
        alg.tell(point, None)

    def test_generation_updates_covariance_and_keeps_dominating_offspring(self):
        alg = OptunaAlgorithm('cmaes', 4, 7)
        initialize(alg)
        for _ in range(4):
            point, details = alg.ask()
            self.assertEqual(details['generation'], 1)
            alg.tell(point, [.5, .5])  # Every offspring dominates every parent.
        # Upstream updates at the NEXT ask, never by adding hidden evaluations.
        self.assertNotIn('mocma:generation:1:elite_ids', study_attrs(alg))
        point, details = alg.ask()
        self.assertEqual(details['generation'], 2)
        attrs = study_attrs(alg)
        self.assertEqual(set(attrs['mocma:generation:1:elite_ids']), {4, 5, 6, 7})
        self.assertFalse(np.allclose(attrs['mocma:trial:4:cov'], np.eye(3)))
        self.assertNotEqual(attrs['mocma:trial:4:sigma'], 1/6)
        alg.tell(point, [.4, .4])

    def test_failed_offspring_does_not_complete_a_generation(self):
        alg = OptunaAlgorithm('cmaes', 4, 7)
        initialize(alg)
        point, _ = alg.ask()
        alg.tell(point, None)
        self.assertIsNone(alg.study.trials[4].values)
        instances = []
        for _ in range(4):
            point, details = alg.ask()
            self.assertEqual(details['generation'], 1)
            instances.append(details['instance'])
            alg.tell(point, [.5, .5])
        self.assertEqual(set(instances), set(range(4)))
        point, details = alg.ask()
        self.assertEqual(details['generation'], 2)
        self.assertNotIn(4, study_attrs(alg)['mocma:generation:1:elite_ids'])
        self.assertEqual(len(alg.study.trials), 10)
        alg.tell(point, [.4, .4])

    def test_upstream_clipping_is_the_actual_evaluated_design(self):
        alg = OptunaAlgorithm('cmaes', 4, 7)
        initialize(alg)
        rng = Mock(wraps=np.random.RandomState(7))
        rng.multivariate_normal.return_value = np.array([-.2, 1.2, 3.4])
        with patch.object(alg.study.sampler._rng, '_rng', rng):
            point, _ = alg.ask()
        # Optuna's transform nudges a float at the high endpoint inward by one
        # representable step; the stored/evaluated design must still agree exactly.
        np.testing.assert_allclose(point, [0., 1., 1.], rtol=0, atol=np.finfo(float).eps)
        alg.tell(point, [.4, .4])
        np.testing.assert_array_equal(list(alg.study.trials[-1].params.values()), point)

    def test_reproducible_with_interleaved_runs_and_external_rng_unchanged(self):
        a, b = [OptunaAlgorithm('cmaes', 4, 42) for _ in range(2)]
        for alg in (a, b):
            initialize(alg)
        external_state = np.random.get_state()
        try:
            for _ in range(25):
                state = np.random.get_state()
                x, _ = a.ask()
                after = np.random.get_state()
                np.testing.assert_array_equal(state[1], after[1])
                self.assertEqual(state[2:], after[2:])
                np.random.random(17)  # Unrelated draws must not change the sampler.
                y, _ = b.ask()
                np.testing.assert_array_equal(x, y)
                self.assertTrue(np.isfinite(x).all())
                self.assertTrue(np.all((x >= 0) & (x <= 1)))
                objectives = [x[0]+x[1]**2, 1-x[0]+x[2]**2]
                a.tell(x, objectives)
                b.tell(y, objectives)
        finally:
            np.random.set_state(external_state)

    def test_all_failed_initialization_recovers_after_a_full_successful_population(self):
        alg = OptunaAlgorithm('cmaes', 4, 42)
        initialize(alg, failed=(0, 1, 2, 3))
        for _ in range(4):
            point, details = alg.ask()
            self.assertEqual(details['proposal'], 'mocma_initial_population')
            self.assertTrue(np.isfinite(point).all())
            alg.tell(point, [.7, .8])
        point, details = alg.ask()
        self.assertEqual(details['proposal'], 'optunahub_mocma')
        self.assertEqual(set(study_attrs(alg)['mocma:generation:0:elite_ids']), {4, 5, 6, 7})
        alg.tell(point, [.5, .6])
        self.assertEqual(len(alg.study.trials), 9)

    def test_pending_protocol_and_response_validation(self):
        alg = OptunaAlgorithm('cmaes', 4, 42)
        with self.assertRaises(ValueError):
            alg.tell([.5]*3, [0., 0.])
        point, _ = alg.ask([.5]*3)
        with self.assertRaises(ValueError):
            alg.ask()
        for wrong_x, wrong_y in [([0]*3, [0, 0]), (point, [np.nan, 0]), (point, [1])]:
            with self.assertRaises(ValueError):
                alg.tell(wrong_x, wrong_y)
        alg.tell(point, None)

    def test_cli_cma_only_odd_budget_stops_in_a_partial_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'run'
            with patch('aemec_benchmark.evaluation.shutil.which', return_value='/mock/tool'), \
                    patch('aemec_opt.case.AemecOpenFoamEvaluator') as adapter:
                adapter.return_value.evaluate.return_value = SimpleNamespace(
                    as_objective_result=lambda: ObjectiveResult((2., 3e-9), {}))
                self.assertEqual(main(['--algorithms', 'cmaes', '--output', str(out),
                                       '--initial', '4', '--budget', '9', '--seeds', '7']), 0)
                self.assertEqual(adapter.return_value.evaluate.call_count, 9)
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['actual_evaluator_calls'], 9)
            self.assertNotIn('cma_sigma', manifest['settings'])
            self.assertEqual(manifest['cmaes_protocol']['initial_sigma'], 1/6)
            self.assertEqual(manifest['settings']['algorithms'], ['cmaes'])
            history = pd.read_csv(out/'seed_7/cmaes/history.csv')
            self.assertEqual(len(history), 9)
            self.assertTrue((history.failed == 0).all())
            self.assertTrue(np.all(np.diff(history.hypervolume) >= -1e-12))
            self.assertIn('OptunaHub MoCmaSampler', (out/'report.md').read_text())

    def test_obsolete_sigma_option_fails_before_evaluating_or_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'run'
            error = io.StringIO()
            with redirect_stderr(error), patch('aemec_benchmark.cli.build_problem') as build:
                self.assertEqual(main(['--output', str(out), '--cma-sigma', '.2']), 2)
                build.assert_not_called()
            self.assertIn('Remove --cma-sigma', error.getvalue())
            self.assertFalse(out.exists())

    def test_unavailable_sampler_fails_before_any_evaluation_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'run'
            args = build_parser().parse_args(['--output', str(out), '--dry-run'])
            problem = build_problem(args)
            with patch('aemec_benchmark.runner.require_mocma', side_effect=ValueError('sampler unavailable')), \
                    patch('aemec_opt.case.AemecOpenFoamEvaluator') as adapter:
                with self.assertRaisesRegex(ValueError, 'sampler unavailable'):
                    run_benchmark(problem, BenchmarkSettings(algorithms=('cmaes',)), out)
                adapter.assert_not_called()
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
