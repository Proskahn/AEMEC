import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ELA.ela import (FEATURE_COUNT, FEATURE_NAMES, INPUTS, OUTPUTS, analyze, features,
                     pareto_mask, prepare, sample)


def example(n=64):
    frame = sample(n)
    u = (frame[INPUTS].to_numpy() - [10, 298.15, .4]) / [90, 55, .4]
    frame[OUTPUTS[0]] = 1.5 + .2*u[:, 0] + .1*u[:, 1]
    frame[OUTPUTS[1]] = 1e-8*(1 + (u[:, 0]-.3)**2 + u[:, 2])
    return frame


class AnalysisTests(unittest.TestCase):
    def test_nested_sampling_and_bounds(self):
        np.testing.assert_array_equal(sample(32).to_numpy(), sample(128).iloc[:32].to_numpy())
        with self.assertRaises(ValueError):
            sample(33)

    def test_failures_excluded_without_penalty(self):
        frame = example()
        frame['status'] = 'complete'
        frame.loc[0, 'status'] = 'failed'
        frame.loc[1, OUTPUTS[1]] = np.nan
        accepted, excluded, x = prepare(frame)
        self.assertEqual(len(accepted), 62)
        self.assertEqual(set(excluded.exclusion_reason), {'unsuccessful_status', 'missing_or_nonfinite_value'})
        self.assertTrue(((x >= 0) & (x <= 1)).all().all())

    def test_duplicate_and_degenerate_design_rejected(self):
        frame = example()
        frame.loc[1, INPUTS] = frame.loc[0, INPUTS]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            prepare(frame)
        frame = example()
        frame['ptl_porosity'] = .6
        with self.assertRaisesRegex(ValueError, 'span'):
            prepare(frame)

    def test_pareto_minimizes_both_and_keeps_ties(self):
        np.testing.assert_array_equal(pareto_mask([[1, 3], [2, 2], [3, 3], [1, 3]]),
                                      [True, True, False, True])

    def test_constant_is_explicitly_skipped(self):
        _, _, x = prepare(example())
        values, diagnostics = features(x, np.ones(len(x)))
        self.assertEqual(len(values), 49)
        self.assertTrue(all(v is None for v in values.values()))
        self.assertTrue(all(d['status'] == 'skipped' for d in diagnostics))

    def test_linear_meta_and_repeatability(self):
        _, _, x = prepare(example())
        y = x.to_numpy() @ np.array([.2, .3, .5])
        a, diagnostics = features(x, y)
        b, _ = features(x, y)
        self.assertFalse([d for d in diagnostics if d['status'] == 'failed'], diagnostics)
        self.assertAlmostEqual(a['ela_meta.lin_simple.adj_r2'], 1.)
        self.assertEqual(a.keys(), b.keys())
        for key in a:
            if a[key] is not None:
                self.assertAlmostEqual(a[key], b[key])

    def test_exact_table_schema_and_pca_against_eigenvalues(self):
        _, _, x = prepare(example(128))
        y = .2*x.iloc[:, 0]**2 + .3*x.iloc[:, 1] + .5*x.iloc[:, 2]
        values, diagnostics = features(x, y)
        self.assertEqual(FEATURE_COUNT, 49)
        self.assertEqual({k: len(v) for k, v in FEATURE_NAMES.items()}, {
            'distribution': 3, 'level': 9, 'meta': 9, 'dispersion': 16,
            'nbc': 5, 'pca': 2, 'ic': 5})
        self.assertEqual(len(values), 49)
        self.assertFalse(any('costs_runtime' in k for k in values))
        self.assertFalse([d for d in diagnostics if d['status'] == 'failed'])
        augmented = np.column_stack([x, y])
        for label, matrix in [('cov', np.cov(augmented, rowvar=False)),
                              ('cor', np.corrcoef(augmented, rowvar=False))]:
            eigenvalues = np.linalg.eigvalsh(matrix)
            self.assertAlmostEqual(values[f'pca.expl_var_PC1.{label}_init'],
                                   eigenvalues[-1]/eigenvalues.sum())
        for prefix in ['ratio_mean', 'ratio_median', 'diff_mean', 'diff_median']:
            for q in ['02', '05', '10', '25']:
                self.assertIsNotNone(values[f'disp.{prefix}_{q}'])
        self.assertGreaterEqual(values['ic.h_max'], 0)
        self.assertLessEqual(values['ic.h_max'], 1)
        self.assertGreaterEqual(values['ic.m0'], 0)
        self.assertLessEqual(values['ic.m0'], 1)

    def test_small_dispersion_subset_is_not_fabricated(self):
        _, _, x = prepare(example(32))
        y = np.arange(len(x), dtype=float)
        values, diagnostics = features(x, y)
        for prefix in ['ratio_mean', 'ratio_median', 'diff_mean', 'diff_median']:
            self.assertIsNone(values[f'disp.{prefix}_02'])
            self.assertIsNotNone(values[f'disp.{prefix}_05'])
        detail = next(d for d in diagnostics if d['family'] == 'dispersion' and 'subset_counts' in d)
        self.assertEqual(detail['subset_counts']['0.02'], 1)
        self.assertEqual(detail['status'], 'partial')
        self.assertEqual(len(values), 49)

    def test_family_failure_keeps_schema_and_other_families(self):
        _, _, x = prepare(example())
        with patch('pflacco.classical_ela_features.calculate_pca', side_effect=ValueError('test failure')):
            values, diagnostics = features(x, x.to_numpy().sum(axis=1))
        self.assertEqual(len(values), 49)
        self.assertTrue(all(values[k] is None for k in FEATURE_NAMES['pca']))
        self.assertIsNotNone(values['ic.h_max'])
        self.assertTrue(any(d['family'] == 'pca' and d['status'] == 'failed' for d in diagnostics))

    def test_random_tour_is_reproducible_without_rng_side_effects(self):
        _, _, x = prepare(example())
        y = x.to_numpy().sum(axis=1)
        state = np.random.get_state()
        a, _ = features(x, y, seed=9, ic_sorting='random')
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        b, _ = features(x, y, seed=9, ic_sorting='random')
        self.assertEqual(a, b)
        c, _ = features(x, y, seed=10, ic_sorting='random')
        self.assertNotEqual([a[k] for k in FEATURE_NAMES['ic']], [c[k] for k in FEATURE_NAMES['ic']])

    def test_end_to_end_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            example().to_csv(root/'input.csv', index=False)
            meta = analyze(root/'input.csv', root/'report', provenance='space-filling')
            self.assertEqual(meta['accepted'], 64)
            self.assertFalse([d for ds in meta['diagnostics'].values() for d in ds if d['status'] == 'failed'])
            for name in ['features.csv', 'excluded.csv', 'pareto.csv', 'pareto.png', 'spearman.csv', 'report.md']:
                self.assertTrue((root/'report'/name).is_file())
            saved = json.loads((root/'report/metadata.json').read_text())
            self.assertEqual(saved['feature_set']['count_per_landscape'], 49)
            self.assertEqual(saved['feature_settings']['ic_sorting'], 'nn')
            table = pd.read_csv(root/'report/features.csv')
            self.assertEqual(len(table), 245)
            self.assertTrue((table.groupby('landscape').size() == 49).all())

    def test_normalization_passed_to_all_feature_families(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = example()
            frame.to_csv(root/'input.csv', index=False)
            with patch('ELA.ela.features', wraps=features) as compute:
                analyze(root/'input.csv', root/'report', scales=[1, 3, 0, 1e-7], ic_sorting='random')
            targets = [(frame.cell_voltage_v.to_numpy()-1)/2,
                       frame.crossover_rate_mol_s.to_numpy()/1e-7]
            targets += [w*targets[0]+(1-w)*targets[1] for w in [.25, .5, .75]]
            self.assertEqual(compute.call_count, 5)
            for call, expected in zip(compute.call_args_list, targets):
                np.testing.assert_allclose(call.args[1], expected)
                self.assertEqual(call.args[3], 'random')


if __name__ == '__main__':
    unittest.main()
