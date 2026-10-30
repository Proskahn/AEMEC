"""Proxy search proposes designs, while CFD alone supplies scored observations."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from aemec_benchmark.algorithms import DEFAULT_ALGORITHMS, sobol_points
from aemec_benchmark.cli import main
from aemec_benchmark.proxy_assets import ProxyAssets, load_proxy_assets
from aemec_benchmark.proxy_search import ELAProxySearch
from aemec_benchmark.metrics import hypervolume_2d
from aemec_opt.engine import ObjectiveResult, OptimizationError


LOWER = np.array([10., 298.15, .4])
UPPER = np.array([100., 353.15, .8])
PROXY_COMPARISON = DEFAULT_ALGORITHMS + ('ela_proxy',)


def write_pair(root):
    paths = []
    for objective in ['cell_voltage_v', 'crossover_rate_mol_s']:
        path = root/objective/'proxy_01/proxy.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            'schema_version': 1, 'landscape': objective, 'tree': {'op': 'x'},
            'input_columns': ['membrane_thickness_um', 'water_inlet_temperature_k', 'ptl_porosity'],
            'input_lower': LOWER.tolist(), 'input_upper': UPPER.tolist(),
            'raw_offset': 0., 'raw_scale': 1.}))
        paths.append(path)
    return paths


class ProxySearchTests(unittest.TestCase):
    def assets(self, model):
        return ProxyAssets((model, model), np.zeros(3), np.ones(3), {})

    def initialize(self, algorithm):
        for point in sobol_points(4, 7):
            returned, _ = algorithm.ask(point)
            np.testing.assert_array_equal(returned, point)
            algorithm.tell(returned, [2., 3e-9])

    def test_proxy_model_changes_proposals_without_using_cfd_values_as_proxy_scores(self):
        positive = ELAProxySearch(7, self.assets(lambda x: np.sum(x, axis=-1)), 16)
        negative = ELAProxySearch(7, self.assets(lambda x: -np.sum(x, axis=-1)), 16)
        for algorithm in (positive, negative):
            self.initialize(algorithm)
            self.assertEqual(algorithm.function_calls, 0)
        low, details = positive.ask()
        high, _ = negative.ask()
        np.testing.assert_array_equal(low, [0, 0, 0])
        np.testing.assert_array_equal(high, [1, 1, 1])
        self.assertEqual(details['proposal'], 'ela_proxy_augmented_tchebycheff')
        positive.tell(low, None)  # A CFD failure still excludes this attempted point.
        other, _ = positive.ask()
        self.assertFalse(np.array_equal(low, other))

    def test_reproducible_independent_of_global_rng_and_preserves_pending_protocol(self):
        algorithms = [ELAProxySearch(42, self.assets(lambda x: np.sum(x, axis=-1)), 16) for _ in range(2)]
        for algorithm in algorithms:
            self.initialize(algorithm)
        for _ in range(3):
            a, da = algorithms[0].ask()
            np.random.default_rng(102).normal(size=100)
            b, db = algorithms[1].ask()
            np.testing.assert_array_equal(a, b)
            self.assertEqual(da, db)
            with self.assertRaises(ValueError):
                algorithms[0].ask()
            with self.assertRaises(ValueError):
                algorithms[0].tell(a+.01, [1, 2])
            algorithms[0].tell(a, [1, 2])
            algorithms[1].tell(b, [90, -50])  # Direct search keeps the expressions frozen.

    def test_invalid_proxy_values_are_not_substituted_for_cfd_responses(self):
        def singular(x):
            if np.any(np.asarray(x) == 0):
                raise ValueError('singularity')
            return np.sum(x, axis=-1)
        algorithm = ELAProxySearch(42, self.assets(singular), 16)
        self.initialize(algorithm)
        point, details = algorithm.ask()
        self.assertGreater(details['proxy_function_calls_total'], 2)
        self.assertLess(details['finite_proxy_candidates'], details['candidate_count'])
        self.assertTrue(np.all(point > 0))
        bad = ELAProxySearch(42, self.assets(lambda x: np.full(len(x), np.nan)), 16)
        self.initialize(bad)
        point, details = bad.ask()
        self.assertTrue(np.isfinite(point).all())
        self.assertIn('fallback', details['proposal'])
        self.assertNotIn('proxy_values_dimensionless', details)

    def test_artifact_domain_must_match_cfd_and_prior_cost_is_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = write_pair(root)
            (root/'manifest.json').write_text(json.dumps({'sample_count': 256, 'sources': {}}))
            assets = load_proxy_assets(paths, LOWER, UPPER)
            self.assertEqual(assets.metadata['training_provenance'][0]['historical_sample_count'], 256)
            with self.assertRaisesRegex(ValueError, 'domains must match'):
                load_proxy_assets(paths, LOWER, UPPER+[1, 0, 0])

    def test_constant_proxy_objectives_use_finite_diverse_proposals(self):
        algorithm = ELAProxySearch(42, self.assets(lambda x: np.full(len(x), 7.)), 16)
        self.initialize(algorithm)
        point, details = algorithm.ask()
        self.assertEqual(details['proxy_search_score'], 0.)
        self.assertEqual(details['proxy_values_dimensionless'], [7., 7.])
        self.assertTrue(np.isfinite(point).all())
        self.assertFalse(any(np.array_equal(point, previous) for previous in algorithm.attempts))

    def test_all_five_algorithms_record_only_cfd_and_charge_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = write_pair(root/'models')
            out = root/'out'
            def evaluate(number, thickness, temperature, porosity):
                if number == 5:
                    raise OptimizationError('deliberate CFD failure')
                result = ObjectiveResult((2.+thickness*.0001, 3e-9+porosity*1e-10), {'source': 'mock CFD'})
                return SimpleNamespace(as_objective_result=lambda: result)

            with patch('aemec_opt.case.AemecOpenFoamEvaluator') as adapter, \
                    patch('aemec_benchmark.evaluation.shutil.which', return_value='/mock/tool'), \
                    redirect_stdout(io.StringIO()):
                adapter.return_value.evaluate.side_effect = evaluate
                self.assertEqual(main(['--algorithms', *PROXY_COMPARISON, '--output', str(out),
                                       '--proxy-files', *map(str, paths), '--budget', '6', '--initial', '4',
                                       '--seeds', '42', '--gp-candidates', '16', '--gp-restarts', '0',
                                       '--proxy-candidates', '16']), 0)
                self.assertEqual(adapter.return_value.evaluate.call_count, 14)
                self.assertEqual(adapter.call_count, 6)  # Shared pool plus five runs.
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['problem']['evaluator'], 'cfd')
            self.assertEqual(manifest['actual_evaluator_calls'], 14)
            self.assertIn('ela_proxy_protocol', manifest)
            self.assertNotIn('models', manifest['problem'])
            lower, upper = np.asarray(manifest['objective_lower']), np.asarray(manifest['objective_upper'])
            for name in PROXY_COMPARISON:
                rows = pd.read_csv(out/f'seed_42/{name}/evaluations.csv')
                self.assertEqual(len(rows), 6)
                self.assertEqual(rows.status.tolist(), ['complete']*5+['failed'])
                valid = rows.iloc[:5]
                np.testing.assert_allclose(valid.cell_voltage_v, 2.+valid.membrane_thickness_um*.0001)
                np.testing.assert_allclose(valid.crossover_rate_mol_s, 3e-9+valid.ptl_porosity*1e-10, atol=1e-20)
                self.assertTrue(rows.iloc[5][['cell_voltage_v', 'crossover_rate_mol_s']].isna().all())
                values = valid[['cell_voltage_v', 'crossover_rate_mol_s']].to_numpy()
                expected = hypervolume_2d((values-lower)/(upper-lower), manifest['reference_normalized'])
                history = pd.read_csv(out/f'seed_42/{name}/history.csv')
                self.assertAlmostEqual(history.iloc[-1].hypervolume, expected)
                self.assertAlmostEqual(history.iloc[-2].hypervolume, expected)
            details = json.loads((out/'seed_42/ela_proxy/details.json').read_text())
            self.assertIn('proxy_values_dimensionless', details[4]['proposal'])
            self.assertIn('ELA proxy search', (out/'report.md').read_text())


if __name__ == '__main__':
    unittest.main()
