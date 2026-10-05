import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ELA.ela import INPUTS, OUTPUTS, LOWER, UPPER, prepare, sample
from ELA.plotting import (conditional_slices, cross_validate, direction, fit,
                          plot_effects, predict)


def responses(x):
    return np.column_stack([1.5+.5*x[:, 0]-.15*x[:, 1]+.2*(x[:, 2]-.4)**2,
                            1e-8*(2-.5*x[:, 0]-.3*x[:, 1]+.1*x[:, 2])])


def dataset():
    frame = sample(128)
    x = (frame[INPUTS].to_numpy()-LOWER)/(np.array(UPPER)-LOWER)
    frame[OUTPUTS] = responses(x)
    return frame


class PlottingTests(unittest.TestCase):
    def test_conditional_curves_and_adjusted_points_recover_known_effects(self):
        frame, _, normalized = prepare(dataset())
        x, y = normalized.to_numpy(), frame[OUTPUTS].to_numpy()
        coefficient = fit(x, y, 'quadratic')
        reference = np.array([.6, .3, .7])
        slices = conditional_slices(x, y, coefficient, 'quadratic', reference, 101)
        for j, (grid, prediction, adjusted) in enumerate(slices):
            check = np.tile(reference, (len(grid), 1))
            check[:, j] = grid
            np.testing.assert_allclose(prediction, responses(check), rtol=1e-10)
            check = np.tile(reference, (len(x), 1))
            check[:, j] = x[:, j]
            np.testing.assert_allclose(adjusted, responses(check), rtol=1e-10)
        self.assertEqual(direction(slices[0][1][:, 0], np.ptp(y[:, 0])), 'increasing')
        self.assertEqual(direction(slices[0][1][:, 1], np.ptp(y[:, 1])), 'decreasing')
        self.assertEqual(direction(slices[1][1][:, 0], np.ptp(y[:, 0])), 'decreasing')
        self.assertEqual(direction(slices[1][1][:, 1], np.ptp(y[:, 1])), 'decreasing')
        self.assertEqual(direction(slices[2][1][:, 0], np.ptp(y[:, 0])), 'non_monotonic')

    def test_cv_distinguishes_linear_from_curved_and_is_reproducible(self):
        frame, _, normalized = prepare(dataset())
        x, y = normalized.to_numpy(), frame[OUTPUTS].to_numpy()
        prediction, ids, scores = cross_validate(x, y, 'quadratic')
        np.testing.assert_allclose(prediction, y, rtol=1e-10)
        self.assertEqual(len(set(ids)), 5)
        again, ids_again, _ = cross_validate(x, y, 'quadratic')
        np.testing.assert_array_equal(prediction, again)
        np.testing.assert_array_equal(ids, ids_again)
        _, _, linear = cross_validate(x, y, 'linear')
        self.assertGreater(linear[0]['cv_rmse'], .001)
        self.assertLess(scores[0]['cv_rmse'], 1e-12)

    def test_interaction_can_reverse_an_effect_at_different_references(self):
        _, _, normalized = prepare(dataset())
        x = normalized.to_numpy()
        y = np.column_stack([1+.5*x[:, 0]+(x[:, 0]-.5)*x[:, 1], 1e-8*(2+x[:, 1])])
        coefficient = fit(x, y, 'quadratic')
        low = conditional_slices(x, y, coefficient, 'quadratic', np.array([.2, .5, .5]), 101)
        high = conditional_slices(x, y, coefficient, 'quadratic', np.array([.8, .5, .5]), 101)
        self.assertEqual(direction(low[1][1][:, 0], np.ptp(y[:, 0])), 'decreasing')
        self.assertEqual(direction(high[1][1][:, 0], np.ptp(y[:, 0])), 'increasing')
        self.assertEqual(direction(low[1][1][:, 1], np.ptp(y[:, 1])), 'increasing')
        self.assertEqual(direction(high[1][1][:, 1], np.ptp(y[:, 1])), 'increasing')

    def test_constant_output_is_supported_without_false_r2(self):
        frame, _, normalized = prepare(dataset())
        y = frame[OUTPUTS].to_numpy()
        y[:, 1] = 1e-8
        prediction, _, metrics = cross_validate(normalized.to_numpy(), y, 'quadratic')
        np.testing.assert_array_equal(prediction[:, 1], y[:, 1])
        self.assertIsNone(metrics[1]['cv_r2'])
        self.assertIsNone(metrics[1]['cv_rmse_over_observed_range'])
        self.assertEqual(direction(prediction[:, 1], 0), 'flat')

    def test_rank_deficient_polynomial_is_rejected(self):
        x = np.random.default_rng(0).random((32, 3))
        x[:, 2] = np.round(x[:, 2])  # x3 and x3 squared are identical.
        with self.assertRaisesRegex(ValueError, 'identify all polynomial'):
            fit(x, responses(x), 'quadratic')

    def test_report_and_physical_unit_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = dataset()
            frame['status'] = 'complete'
            frame.loc[0, 'status'] = 'failed'
            frame.to_csv(root/'evaluated.csv', index=False)
            reference = [50, 320, .6]
            meta = plot_effects(root/'evaluated.csv', root/'plots', reference=reference)
            self.assertEqual(meta['accepted'], 127)
            self.assertEqual(meta['excluded'], 1)
            self.assertEqual(list(meta['reference'].values()), reference)
            for name in ['raw_scatter', 'conditional_effects', 'objective_directions', 'model_validation']:
                for extension in ['png', 'pdf']:
                    self.assertGreater((root/'plots'/f'{name}.{extension}').stat().st_size, 1000)
            table = pd.read_csv(root/'plots/conditional_curves.csv')
            self.assertEqual(len(table), 6*201)
            membrane = table[(table.input == INPUTS[0]) & (table.objective == OUTPUTS[1])]
            self.assertGreater(membrane.input_value.min(), 10)
            self.assertLess(membrane.input_value.max(), 100)
            self.assertLess(membrane.predicted_value.max(), 1e-7)
            summary = pd.read_csv(root/'plots/effect_summary.csv')
            self.assertEqual(len(summary), 6)
            json.loads((root/'plots/plot_metadata.json').read_text())

    def test_mixed_operating_conditions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = dataset()
            frame['target_current_density_a_m2'] = 10000
            frame.loc[0, 'target_current_density_a_m2'] = 20000
            frame.to_csv(root/'mixed.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'one consistent'):
                plot_effects(root/'mixed.csv', root/'plots')
            self.assertFalse((root/'plots').exists())


if __name__ == '__main__':
    unittest.main()
