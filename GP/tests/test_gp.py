"""Small synthetic cases check CV isolation, physical units, and saved-model use."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from scipy.stats import qmc

from GP.models import INPUTS, OUTPUTS, cross_validate, load_data, predict_saved
from GP.train_gp import train


def sample_frame(n=32):
    unit = qmc.Sobol(3, scramble=True, seed=7).random_base2(int(np.log2(n)))
    frame = pd.DataFrame(qmc.scale(unit, [10, 298.15, .4], [100, 353.15, .8]), columns=INPUTS)
    frame['sample_id'] = np.arange(n)
    frame['status'] = 'complete'
    frame['ptl_side'] = 'anode'
    frame['target_current_density_a_m2'] = 10000.
    frame[OUTPUTS[0]] = 1.7+.3*unit[:, 0]-.1*unit[:, 1]+.04*unit[:, 2]
    frame[OUTPUTS[1]] = 1e-9*(1+2/(.3+unit[:, 0])+.2*unit[:, 1]*unit[:, 2])
    return frame


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.csv = Path(self.directory.name)/'evaluated.csv'
        self.frame = sample_frame()

    def test_preserve_successful_samples_and_audit_failures(self):
        bad = self.frame.iloc[:3].copy()
        bad['sample_id'] += 100
        bad.loc[0, 'status'] = 'failed'
        bad.loc[1, OUTPUTS[1]] = np.inf
        bad.loc[2, INPUTS[2]] = 1.1
        pd.concat([self.frame, bad]).to_csv(self.csv, index=False)
        accepted, excluded = load_data(self.csv, expected_samples=32)
        self.assertEqual(len(accepted), 32)
        self.assertEqual(set(excluded.exclusion_reason),
                         {'unsuccessful_status', 'missing_or_nonfinite_value', 'invalid_physical_value'})
        self.assertAlmostEqual(accepted[OUTPUTS[1]].iloc[0], self.frame[OUTPUTS[1]].iloc[0], places=20)

    def test_missing_evaluations_do_not_silently_reduce_dataset(self):
        self.frame.to_csv(self.csv, index=False)
        with self.assertRaisesRegex(ValueError, 'Expected 256 valid samples, found 32'):
            load_data(self.csv)

    def test_duplicate_designs_and_mixed_operating_conditions_are_rejected(self):
        duplicate = self.frame.copy()
        duplicate.loc[1, INPUTS] = duplicate.loc[0, INPUTS]
        duplicate.to_csv(self.csv, index=False)
        with self.assertRaisesRegex(ValueError, 'Duplicate input designs'):
            load_data(self.csv, 32)
        mixed = self.frame.copy()
        mixed.loc[1, 'target_current_density_a_m2'] = 20000.
        mixed.to_csv(self.csv, index=False)
        with self.assertRaisesRegex(ValueError, 'consistent operating setup'):
            load_data(self.csv, 32)


class CrossValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = sample_frame()
        cls.oof, cls.metrics, cls.fold_metrics, cls.details = cross_validate(
            cls.frame, folds=4, seed=11, restarts=0, progress=lambda *args, **kwargs: None)

    def test_every_sample_held_out_once_per_objective(self):
        for objective in OUTPUTS:
            seen = []
            for fit in self.details:
                if fit['objective'] != objective:
                    continue
                self.assertFalse(set(fit['train_indices']) & set(fit['test_indices']))
                self.assertEqual(len(fit['train_indices'])+len(fit['test_indices']), len(self.frame))
                seen += fit['test_indices']
                self.assertTrue((self.oof.loc[fit['test_indices'], 'cv_fold'] == fit['fold']).all())
            self.assertEqual(sorted(seen), list(range(len(self.frame))))

    def test_normalization_uses_training_fold_only(self):
        for fit in self.details:
            training = self.frame.iloc[fit['train_indices']]
            np.testing.assert_allclose(fit['input_mean'], training[INPUTS].mean())
            np.testing.assert_allclose(fit['input_scale'], training[INPUTS].std(ddof=0))
            self.assertAlmostEqual(fit['output_mean'], training[fit['objective']].mean(), places=20)
            self.assertAlmostEqual(fit['output_scale'], training[fit['objective']].std(ddof=0), places=20)
            self.assertFalse(np.allclose(fit['input_mean'], self.frame[INPUTS].mean()))

    def test_metrics_and_predictions_are_held_out_and_in_physical_units(self):
        for objective in OUTPUTS:
            mean = self.oof[f'{objective}_gp_mean'].to_numpy()
            std = self.oof[f'{objective}_gp_std'].to_numpy()
            self.assertTrue(np.isfinite(mean).all())
            self.assertTrue((std > 0).all())
            metric = self.metrics.loc[(self.metrics.objective == objective) & (self.metrics.model == 'gp')].iloc[0]
            self.assertAlmostEqual(metric.rmse, np.sqrt(np.mean((mean-self.frame[objective])**2)), places=20)
            self.assertGreater(metric.r2, .8)
        self.assertLess(self.oof[f'{OUTPUTS[1]}_gp_mean'].abs().max(), 1e-8)
        self.assertLess(self.oof[f'{OUTPUTS[1]}_gp_std'].max(), 1e-8)
        linear = self.metrics.loc[(self.metrics.objective == OUTPUTS[0]) & (self.metrics.model == 'linear_baseline')]
        self.assertLess(linear.iloc[0].rmse, 1e-12)


class SavedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.frame = sample_frame()
        cls.source = cls.root/'evaluated.csv'
        cls.frame.to_csv(cls.source, index=False)
        cls.output = cls.root/'results'
        with contextlib.redirect_stdout(io.StringIO()):
            cls.metadata = train(cls.source, cls.output, expected_samples=32, folds=2, restarts=0)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_complete_bundle_and_reload(self):
        expected = ['metadata.json', 'report.md', 'cross_validation.png', 'cross_validation.pdf',
                    'training_data.csv', 'excluded.csv', 'cv_predictions.csv', 'cv_metrics.csv', 'cv_fold_metrics.csv']
        for name in expected:
            self.assertGreater((self.output/name).stat().st_size, 0)
        self.assertEqual(json.loads((self.output/'metadata.json').read_text())['sample_count'], 32)
        self.assertFalse(list(self.root.glob('.gp-training-*')))
        # Interior designs avoid CSV roundoff at the observed extrema.
        design = self.frame.iloc[:4][INPUTS].copy()
        design.iloc[:, :] = self.frame[INPUTS].mean().to_numpy()
        predicted = predict_saved(self.output, design, self.metadata)
        for objective in OUTPUTS:
            model = joblib.load(self.output/f'{objective}.joblib')
            direct, direct_std = model.predict(design.to_numpy(), return_std=True)
            np.testing.assert_allclose(predicted[f'{objective}_gp_mean'], direct, rtol=1e-12, atol=0)
            np.testing.assert_allclose(predicted[f'{objective}_gp_std'], direct_std, rtol=1e-12, atol=0)
            self.assertEqual(len(model.named_steps['gp'].X_train_), 32)

    def test_extrapolation_requires_explicit_option(self):
        design = self.frame.iloc[:1][INPUTS].copy()
        design.loc[0, INPUTS[0]] = 150.
        with self.assertRaisesRegex(ValueError, 'beyond observed training ranges'):
            predict_saved(self.output, design, self.metadata)
        predicted = predict_saved(self.output, design, self.metadata, allow_extrapolation=True)
        self.assertTrue(predicted.outside_observed_input_ranges.iloc[0])

    def test_existing_bundle_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError, 'not empty'):
            train(self.source, self.output, expected_samples=32, restarts=0)


if __name__ == '__main__':
    unittest.main()
