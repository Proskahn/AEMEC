import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ELA.ela import INPUTS, OUTPUTS, analyze
from ELA.generate_proxies import main, parser
from ELA.proxy_functions import ARITY, Node, ProxyFunction, evaluate_tree, parse_tree
from ELA.proxy_generation import (FEATURES, Candidate, LandscapeScorer, artifact,
                                  crossover, distance_scales, evolve, export_proxy,
                                  feature_distance, load_analysis, make_subsets,
                                  mutate, random_tree, run, select_features,
                                  validation_diagnostics)
from ELA.tests.test_ela import example


class ExpressionTests(unittest.TestCase):
    def test_paper_operators_and_vector_axis(self):
        self.assertEqual(len(ARITY), 20)
        x = np.array([[.2, .4, .8], [.3, .5, .7]])
        expected = {'neg': -x, 'rec': 1/x, 'mult10': 10*x, 'square': x*x,
                    'abs': abs(x), 'sqrt': np.sqrt(x), 'exp': np.exp(x),
                    'ln': np.log(x), 'sin': np.sin(2*np.pi*x), 'cos': np.cos(2*np.pi*x),
                    'round': np.rint(x), 'cum': np.cumsum(x, axis=1),
                    'sum': x.sum(axis=1, keepdims=True), 'mean': x.mean(axis=1, keepdims=True),
                    'prod': x.prod(axis=1, keepdims=True), 'max': x.max(axis=1, keepdims=True)}
        for op, values in expected.items():
            np.testing.assert_allclose(evaluate_tree(Node(op, (Node('x'),)), x), values.sum(axis=1))
        for op, values in {'add': x+2, 'sub': x-2, 'mul': x*2, 'div': x/2}.items():
            tree = Node(op, (Node('x'), Node('constant', value=2.)))
            np.testing.assert_allclose(evaluate_tree(tree, x), values.sum(axis=1))
        tree = Node('sum', (Node('mul', (Node('x'), Node('index'))),))
        np.testing.assert_allclose(evaluate_tree(tree, x), x @ np.arange(1, 4))
        self.assertEqual(evaluate_tree(tree, x[:1])[0], evaluate_tree(tree, x)[0])

    def test_serialization_rejects_executable_or_invalid_nodes(self):
        tree = Node('mul', (Node('x'), Node('constant', value=.25)))
        self.assertEqual(parse_tree(json.loads(json.dumps(tree.to_dict()))), tree)
        for invalid in [{'op': '__import__', 'children': []}, {'op': 'add', 'children': []},
                        {'op': 'constant', 'value': float('nan')}]:
            with self.assertRaises(ValueError):
                parse_tree(invalid)

    def test_variation_is_reproducible_and_bounded(self):
        rng = np.random.default_rng(42)
        trees = [random_tree(rng, 3, 6, 50) for _ in range(20)]
        rng2 = np.random.default_rng(42)
        self.assertEqual(trees, [random_tree(rng2, 3, 6, 50) for _ in range(20)])
        for a, b in zip(trees, trees[1:]):
            for child in (*crossover(a, b, rng, 6, 50), mutate(a, rng, 6, 50)):
                self.assertLessEqual(child.depth, 6)
                self.assertLessEqual(child.size, 50)

    def test_exported_callable_and_physical_input_contract(self):
        candidate = Candidate(Node('sum', (Node('x'),)), 0., 0., 3.)
        meta = {'input_lower': [10, 298.15, .4], 'input_upper': [100, 353.15, .8]}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/'model'
            model = export_proxy(folder, candidate, OUTPUTS[0], meta, 42)
            function = ProxyFunction(model)
            self.assertAlmostEqual(function([55, 325.65, .6]), .5)
            np.testing.assert_allclose(function([meta['input_lower'], meta['input_upper']]), [0, 1])
            spec = importlib.util.spec_from_file_location('test_generated_proxy', folder/'proxy.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertAlmostEqual(module.objective([55, 325.65, .6]), .5)
            for invalid in [[9, 300, .5], [20, np.nan, .5], [10, 20]]:
                with self.assertRaises(ValueError):
                    function(invalid)


class MatchingTests(unittest.TestCase):
    def test_subsets_are_distinct_designs_reproducible_and_fractional(self):
        subsets = make_subsets(128)
        self.assertEqual(len(subsets), 5)
        self.assertEqual(len(subsets[0]), 102)
        for a, b in zip(subsets, make_subsets(128)):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(len(set(a)), len(a))
        for kwargs in [{'n': 32}, {'n': 128, 'repeats': 1}, {'n': 128, 'fraction': 1.}]:
            with self.assertRaises(ValueError):
                make_subsets(**kwargs)

    def test_fixed_feature_mask_and_per_feature_wasserstein(self):
        target = np.full((5, len(FEATURES)), np.nan)
        target[:, 0] = np.arange(5)
        target[:, 1] = -np.arange(5)
        target[:, 2] = 7.  # constant feature remains a constraint
        retained, excluded = select_features(target)
        self.assertEqual(retained, [0, 2])
        self.assertIn('correlated', excluded[FEATURES[1]])
        candidate = target.copy()
        candidate[:, 0] += 2
        candidate[:, 2] += 4
        score, components = feature_distance(target, candidate, retained, np.ones(2))
        self.assertEqual(score, 3.)
        np.testing.assert_array_equal(components, [2, 4])
        # Flattening all feature entries would silently match unlike features.
        a, b = np.array([[0., 100.]]), np.array([[100., 0.]])
        self.assertEqual(feature_distance(a, b, [0, 1], [1, 1])[0], 100.)
        candidate[0, 2] = np.nan
        self.assertTrue(np.isinf(feature_distance(target, candidate, retained, np.ones(2))[0]))
        self.assertTrue(np.isfinite(distance_scales(target, retained, 'target-std')).all())

    def test_invalid_constant_and_boundary_singular_candidates_are_penalized(self):
        x = np.random.default_rng(1).uniform(.1, .9, size=(64, 3))
        target = np.ones((2, len(FEATURES)))
        scorer = LandscapeScorer(x, make_subsets(64, repeats=2), target, [0], [1.], 42, 'nn', 42)
        for tree in [Node('constant', value=1.), Node('rec', (Node('x'),)),
                     Node('exp', (Node('constant', value=1000.),))]:
            self.assertTrue(np.isinf(scorer(tree).fitness))
        self.assertEqual(scorer.evaluations, 0)

    def test_undefined_target_validation_is_reported_without_mask_changes(self):
        target = np.ones((5, len(FEATURES)))
        candidate = target.copy()
        target[2, 0] = np.nan
        retained = [0, 1]
        score, _ = feature_distance(target, candidate, retained, [1., 1.])
        self.assertTrue(np.isinf(score))
        self.assertEqual(validation_diagnostics(target, candidate, retained), {
            'undefined_target_features': {FEATURES[0]: [2]}, 'undefined_proxy_features': {}})

    def test_fitness_uses_features_and_preserves_full_pool_normalization(self):
        x = np.random.default_rng(7).uniform(0, 1, size=(64, 3))
        target = np.ones((2, len(FEATURES)))
        scorer = LandscapeScorer(x, make_subsets(64, repeats=2), target, [0], [1.], 42, 'nn', 42)
        tree = Node('sum', (Node('x'),))
        with patch('ELA.proxy_generation.feature_distribution', return_value=target) as compute:
            result = scorer(tree)
            self.assertEqual(result.fitness, 0.)
            y = compute.call_args.args[1]
            np.testing.assert_allclose(y, (x.sum(1)-x.sum(1).min())/np.ptp(x.sum(1)))
            scorer(tree)
            self.assertEqual(compute.call_count, 1)

    def test_small_end_to_end_real_ela_search(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            example(64).to_csv(root/'evaluated.csv', index=False)
            analyze(root/'evaluated.csv', root/'analysis', provenance='space-filling')
            args = parser().parse_args(['fit', str(root/'analysis'), '--output', str(root/'proxies'),
                                       '--landscapes', OUTPUTS[0], '--population', '4', '--generations', '0',
                                       '--min-depth', '1', '--max-depth', '2', '--repeats', '2', '--top-k', '1'])
            # Fix the initial population to known, distinct analytical landscapes;
            # all ELA feature extraction, matching, export and validation are real.
            starters = [Node('sum', (Node('x'),)), Node('sum', (Node('square', (Node('x'),)),)),
                        Node('sum', (Node('mul', (Node('x'), Node('index'))),)),
                        Node('mean', (Node('sqrt', (Node('x'),)),))]
            with patch('ELA.proxy_generation.random_tree', side_effect=starters):
                summaries = run(args)
            self.assertEqual(len(summaries), 1)
            manifest = json.loads((root/'proxies/manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['sample_count'], 64)
            function_path = root/'proxies'/summaries[0]['artifact']
            self.assertTrue(np.isfinite(ProxyFunction(function_path)([50, 325, .6])))
            features_path = function_path.parent/'feature_comparison.csv'
            comparison = pd.read_csv(features_path)
            self.assertEqual(len(comparison), 49)
            self.assertEqual(main(['evaluate', str(function_path), '--input', str(root/'evaluated.csv'),
                                   '--output', str(root/'values.csv')]), 0)
            self.assertEqual(len(pd.read_csv(root/'values.csv')), 64)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                run(args)
            table = pd.read_csv(root/'analysis/features.csv')
            table.loc[0, 'value'] += 123
            table.to_csv(root/'analysis/features.csv', index=False)
            args.output = root/'stale'
            with self.assertRaisesRegex(ValueError, 'does not match'):
                run(args)


if __name__ == '__main__':
    unittest.main()
