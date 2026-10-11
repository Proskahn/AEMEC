import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixed_bo.acquisition import expected_hypervolume_improvement, pareto_front, suggest, not_previously_evaluated
from mixed_bo.cli import new_bundle, synthetic_evaluations
from mixed_bo.data import CATEGORICAL, CONTINUOUS, OBJECTIVES, DESIGN, read_catalog, read_evaluations, initial_design
from mixed_bo.representation import DescriptorVAE, fit_representation, feature_matrix, load_representation, standardized
from mixed_bo.surrogate import train_surrogates, load_surrogates, splits_for, predict

ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def setUp(self):
        self.catalog = read_catalog(ROOT/'examples/materials.demo.json', True)

    def test_unfilled_and_synthetic_catalogs_require_explicit_action(self):
        with self.assertRaisesRegex(ValueError, 'human-defined'):
            read_catalog(ROOT/'config/materials.template.json')
        with self.assertRaisesRegex(ValueError, '--allow-demo'):
            read_catalog(ROOT/'examples/materials.demo.json')

    def test_design_covers_only_all_nine_valid_pairs(self):
        frame = initial_design(self.catalog, per_combination=3, seed=9)
        self.assertEqual(len(frame), 27)
        self.assertEqual(frame.groupby(CATEGORICAL).size().tolist(), [3]*9)
        self.assertTrue(frame[OBJECTIVES].isna().all().all())
        self.assertTrue((frame.status == 'pending').all())
        pd.testing.assert_frame_equal(frame, initial_design(self.catalog, 3, 9))

    def test_tiny_physical_descriptor_units_do_not_erase_information(self):
        values = np.array([[1.], [2.], [4.]])
        np.testing.assert_allclose(standardized(values)[0], standardized(values*1e-18)[0], rtol=1e-14)
        np.testing.assert_array_equal(standardized(np.ones((3, 1)))[0], np.zeros((3, 1)))

    def test_missing_real_labels_and_duplicate_mixed_designs_rejected(self):
        frame = synthetic_evaluations(initial_design(self.catalog, 2), self.catalog)
        with tempfile.TemporaryDirectory() as name:
            source = Path(name)/'data.csv'
            frame.drop(columns=CATEGORICAL).to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, 'actual evaluated materials'):
                read_evaluations(source, self.catalog)
            frame.loc[1, DESIGN] = frame.loc[0, DESIGN]
            frame.to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, 'Duplicate complete mixed designs'):
                read_evaluations(source, self.catalog)
            frame.loc[1, 'membrane_type'] = 'Membrane Type_2'
            frame.to_csv(source, index=False)
            accepted, _ = read_evaluations(source, self.catalog)
            self.assertEqual(len(accepted), len(frame))

    def test_category_holdout_keeps_groups_separate(self):
        frame = initial_design(self.catalog, 2)
        for mode in ['combination', *CATEGORICAL]:
            splits = splits_for(frame, mode, folds=5, seed=42)
            seen = []
            for train, test in splits:
                columns = CATEGORICAL if mode == 'combination' else [mode]
                groups = lambda indices: set(map(tuple, frame.iloc[indices][columns].to_numpy()))
                self.assertFalse(groups(train) & groups(test))
                seen.extend(test)
            self.assertEqual(sorted(seen), list(range(len(frame))))


class AcquisitionTests(unittest.TestCase):
    @staticmethod
    def hypervolume(points, reference):
        # Independent rectangle-union sweep for deterministic numerical checks.
        inside = np.asarray(points).reshape(-1, 2)
        inside = inside[np.all(inside < reference, axis=1)]
        if len(inside) == 0:
            return 0.
        edges = sorted(set(inside[:, 0]) | {reference[0]})
        total = 0.
        for left, right in zip(edges[:-1], edges[1:]):
            ceiling = inside[inside[:, 0] <= left, 1].min()
            total += (right-left)*(reference[1]-ceiling)
        return total

    def test_exact_improvement_and_minimization_direction(self):
        observed = np.array([[.2, .8], [.5, .5], [.8, .2], [.9, .9], [.5, .5]])
        reference = np.array([1., 1.])
        candidates = np.array([[.1, .1], [.6, .6], [1.1, .1], [.2, .8], [.4, .4], [-.1, .6]])
        actual = expected_hypervolume_improvement(candidates, np.zeros_like(candidates), observed, reference)
        baseline = self.hypervolume(observed, reference)
        expected = [self.hypervolume(np.vstack([observed, row]), reference)-baseline for row in candidates]
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        self.assertGreater(actual[0], actual[1])
        self.assertAlmostEqual(actual[1], 0.)

    def test_analytic_expectation_matches_monte_carlo(self):
        observed = np.array([[.25, .75], [.7, .3]])
        reference = np.array([1., 1.])
        mean, std = np.array([[.45, .5]]), np.array([[.15, .2]])
        expected = expected_hypervolume_improvement(mean, std, observed, reference)[0]
        draws = np.random.default_rng(2).normal(mean[0], std[0], size=(30000, 2))
        baseline = self.hypervolume(observed, reference)
        values = np.array([self.hypervolume(np.vstack([observed, draw]), reference)-baseline for draw in draws])
        self.assertLess(abs(expected-values.mean()), 5*values.std()/np.sqrt(len(values)))
        self.assertEqual(len(pareto_front(observed)), 2)

    def test_empty_front_and_duplicate_exclusion_respect_categories(self):
        got = expected_hypervolume_improvement(np.array([[.2, .3]]), np.zeros((1, 2)), [], [1., 1.])
        np.testing.assert_allclose(got, [.8*.7])
        catalog = read_catalog(ROOT/'examples/materials.demo.json', True)
        design = initial_design(catalog, 2)
        candidates = design.iloc[:2].copy()
        candidates.loc[candidates.index[1], CONTINUOUS] = candidates.iloc[0][CONTINUOUS].to_numpy()
        candidates.loc[candidates.index[1], 'membrane_type'] = 'Membrane Type_2'
        remaining = not_previously_evaluated(candidates, design.iloc[:1])
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining.membrane_type.iloc[0], 'Membrane Type_2')


class RepresentationAndGPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.catalog = read_catalog(ROOT/'examples/materials.demo.json', True)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.rep = fit_representation(cls.catalog, cls.root/'vae', epochs=600, seed=42)
            cls.frame = synthetic_evaluations(initial_design(cls.catalog, 2), cls.catalog)
            cls.source = cls.root/'observations.csv'
            cls.frame.to_csv(cls.source, index=False)
            cls.model_dir = cls.root/'gp'
            cls.model_dir.mkdir()
            cls.metadata = train_surrogates(cls.source, cls.rep, cls.model_dir, folds=3, restarts=0)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_stochastic_vae_and_both_descriptor_heads_propagate_gradients(self):
        model = DescriptorVAE(3, 1, 1).double()
        identity = torch.eye(3, dtype=torch.double)
        torch.manual_seed(1)
        first = model(identity)
        second = model(identity)
        self.assertFalse(torch.equal(first[1], second[1]))
        for index in [1, 2]:
            model.zero_grad()
            output = model(identity)
            output[index].square().mean().backward()
            self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0.)
            self.assertTrue(torch.isfinite(model.encoder[0].weight.grad).all())

    def test_mean_embeddings_are_fixed_and_reconstruct_descriptors(self):
        loaded = load_representation(self.root/'vae')
        before = feature_matrix(self.frame, self.rep)
        torch.manual_seed(999)
        np.testing.assert_array_equal(before, feature_matrix(self.frame, loaded))
        self.assertEqual(before.shape, (18, 7))
        for name in CATEGORICAL:
            details = self.rep['variables'][name]['diagnostics']
            self.assertEqual(details['identity_accuracy_at_mean'], 1.)
            self.assertLess(details['geometry_mse_standardized_at_mean'], .15)
            self.assertLess(details['physics_mse_standardized_at_mean'], .15)
        self.assertEqual(feature_matrix(self.frame, self.rep, 'onehot').shape[1], 9)
        self.assertEqual(feature_matrix(self.frame, self.rep, 'descriptors').shape[1], 7)

    def test_human_descriptor_changes_influence_the_learned_representation(self):
        altered = copy.deepcopy(self.catalog)
        altered['variables']['anode_pte']['descriptors']['AnodePTE_2']['geometry'] = [.95]
        with contextlib.redirect_stdout(io.StringIO()):
            changed = fit_representation(altered, self.root/'altered', epochs=600, seed=42)
        self.assertFalse(np.allclose(self.rep['variables']['anode_pte']['means'], changed['variables']['anode_pte']['means']))
        np.testing.assert_array_equal(self.rep['variables']['membrane_type']['means'], changed['variables']['membrane_type']['means'])

    def test_cv_excludes_held_out_targets_and_models_reload(self):
        data = pd.read_csv(self.source)
        for detail in self.metadata['cv_fits']:
            train = detail['train_indices']
            test = detail['test_indices']
            self.assertFalse(set(train) & set(test))
            np.testing.assert_allclose(detail['output_mean'], data.iloc[train][detail['objective']].mean(), atol=0)
        models, rep, info = load_surrogates(self.model_dir)
        mean, std = predict(models, self.frame, rep, info['encoding'])
        _, latent_std = predict(models, self.frame, rep, info['encoding'], latent_function=True)
        self.assertTrue(np.isfinite(mean).all())
        self.assertTrue((std >= latent_std).all())
        self.assertLess(abs(mean[:, 1]).max(), 1e-8)
        self.assertTrue((std >= 0).all())
        for model in models:
            self.assertEqual(len(model.X_train_), 18)

    def test_suggestions_stay_on_valid_material_pairs_and_bounds(self):
        output = self.root/'suggestion'
        output.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            result = suggest(self.model_dir, output, candidates_per_combination=16)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[OBJECTIVES].isna().all().all())
        for name in CATEGORICAL:
            self.assertIn(result[name].iloc[0], self.catalog['variables'][name]['types'])
        self.assertGreaterEqual(result.ehvi_normalized.iloc[0], 0.)
        scores = pd.read_csv(output/'candidate_scores.csv')
        self.assertEqual(len(scores.groupby(CATEGORICAL)), 9)
        self.assertEqual(len(not_previously_evaluated(result, self.frame)), 1)
        self.assertEqual(result.status.iloc[0], 'pending')

        pending = self.root/'pending.csv'
        result['sample_id'] = 999
        result.to_csv(pending, index=False)
        other_output = self.root/'suggestion_with_pending'
        other_output.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            other = suggest(self.model_dir, other_output, candidates_per_combination=16, pending=pending)
        self.assertEqual(other.sample_id.iloc[0], 1000)
        self.assertEqual(len(not_previously_evaluated(other, result)), 1)

    def test_modified_representation_is_rejected(self):
        temp = self.root/'tampered'
        temp.mkdir()
        rep = copy.deepcopy(self.rep)
        rep['variables']['anode_pte']['means'][0][0] += 1
        (temp/'representation.json').write_text(json.dumps(rep))
        (temp/'metadata.json').write_text(json.dumps(self.metadata))
        with self.assertRaisesRegex(ValueError, 'representation changed'):
            load_surrogates(temp)

    def test_failed_bundle_cleans_up_and_existing_output_is_preserved(self):
        output = self.root/'incomplete'
        with self.assertRaises(ValueError):
            with new_bundle(output) as work:
                (work/'partial').write_text('partial')
                raise ValueError('failed')
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob('.mixed-bo-*')))
        with self.assertRaisesRegex(ValueError, 'already exists'):
            with new_bundle(self.model_dir):
                pass


if __name__ == '__main__':
    unittest.main()
