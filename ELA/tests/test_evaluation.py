import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from ELA.ela import sample
from ELA.evaluation import ROOT, INPUTS, add_parser, load_design, lock, run_evaluation

sys.path.insert(0, str(ROOT/'opt'))
from aemec_opt.engine import ObjectiveResult, OptimizationError


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root/'source'
        self.source.mkdir()
        (self.source/'model').write_text('test case')
        self.design = self.root/'design.csv'
        sample(4).to_csv(self.design, index=False)
        self.output = self.root/'results'
        self.work = self.root/'work'
        self.executable_check = patch('ELA.evaluation.check_executables').start()
        self.addCleanup(patch.stopall)
        self.factory = patch('aemec_opt.case.AemecOpenFoamEvaluator').start()
        self.factory.return_value.evaluate.return_value = SimpleNamespace(
            as_objective_result=lambda: ObjectiveResult((1.8, 1e-8), {'source': 'mock'}))

    def args(self, *extra):
        parser = argparse.ArgumentParser()
        add_parser(parser.add_subparsers(dest='command'))
        return parser.parse_args(['evaluate', str(self.design), '--case', str(self.source),
                                  '--output-dir', str(self.output), '--work-dir', str(self.work), *extra])

    def run_batch(self, *extra):
        with redirect_stdout(io.StringIO()):
            return run_evaluation(self.args(*extra))

    def records(self):
        return json.loads((self.output/'checkpoint.json').read_text())['rows']

    def test_exact_designs_pilot_and_resume(self):
        self.assertEqual(self.run_batch('--limit', '1'), 0)
        self.assertEqual([r['status'] for r in self.records()], ['complete', 'pending', 'pending', 'pending'])
        self.assertEqual(self.run_batch(), 0)
        expected = load_design(self.design)
        calls = self.factory.return_value.evaluate.call_args_list
        self.assertEqual(len(calls), 4)
        for call, row in zip(calls, expected):
            self.assertEqual(call.args, (row['sample_id'], *(row[k] for k in INPUTS)))
        self.assertEqual(self.run_batch(), 0)
        self.assertEqual(self.factory.return_value.evaluate.call_count, 4)
        self.assertTrue((pd.read_csv(self.output/'evaluated.csv').status == 'complete').all())
        self.assertEqual((self.source/'model').read_text(), 'test case')

    def test_failure_and_explicit_retry_retain_attempt_logs(self):
        success = self.factory.return_value.evaluate.return_value
        self.factory.return_value.evaluate.side_effect = [OptimizationError('unstable'), success]
        self.assertEqual(self.run_batch('--limit', '2'), 1)
        original = self.records()[0]['solver_log']
        self.assertIsNone(self.records()[0]['cell_voltage_v'])
        self.factory.return_value.evaluate.side_effect = None
        self.assertEqual(self.run_batch(), 1)  # Failed row remains excluded, other rows finish.
        self.assertEqual(self.factory.return_value.evaluate.call_count, 4)
        self.assertEqual(self.run_batch('--retry-failed'), 0)
        self.assertEqual(self.factory.return_value.evaluate.call_count, 5)
        retried = self.records()[0]
        self.assertEqual(retried['attempts'], 2)
        self.assertNotEqual(original, retried['solver_log'])
        self.assertEqual(retried['failure_reason'], '')

    def test_interrupt_is_checkpointed_and_retried_on_resume(self):
        success = self.factory.return_value.evaluate.return_value
        self.factory.return_value.evaluate.side_effect = [success, KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt):
            self.run_batch()
        self.assertEqual([r['status'] for r in self.records()], ['complete', 'interrupted', 'pending', 'pending'])
        self.factory.return_value.evaluate.side_effect = None
        self.assertEqual(self.run_batch(), 0)
        self.assertEqual([r['attempts'] for r in self.records()], [1, 2, 1, 1])

    def test_failure_limit_stops_batch_without_replacing_points(self):
        self.factory.return_value.evaluate.side_effect = OptimizationError('mesh failed')
        self.assertEqual(self.run_batch(), 1)
        self.assertEqual(self.factory.return_value.evaluate.call_count, 3)
        self.assertEqual([r['status'] for r in self.records()], ['failed']*3+['pending'])

    def test_resume_rejects_changed_physics_or_design(self):
        self.run_batch('--limit', '1')
        with self.assertRaisesRegex(ValueError, 'Resume settings differ'):
            self.run_batch('--ptl-side', 'cathode')
        (self.source/'model').write_text('modified case')
        with self.assertRaisesRegex(ValueError, 'Resume settings differ'):
            self.run_batch()
        (self.source/'model').write_text('test case')
        sample(4, seed=9).to_csv(self.design, index=False)
        with self.assertRaisesRegex(ValueError, 'Resume settings differ'):
            self.run_batch()
        self.assertEqual(self.factory.return_value.evaluate.call_count, 1)

    def test_dry_run_does_not_write_or_execute(self):
        self.assertEqual(self.run_batch('--dry-run'), 0)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.work.exists())
        self.factory.assert_not_called()
        self.executable_check.assert_not_called()

    def test_owned_scratch_and_output_cannot_be_overwritten(self):
        (self.work/'case').mkdir(parents=True)
        important = self.work/'case/important'
        important.write_text('keep')
        with self.assertRaisesRegex(ValueError, 'ownership marker'):
            self.run_batch()
        self.assertEqual(important.read_text(), 'keep')
        (self.output/'unrelated.csv').write_text('keep too')
        with self.assertRaisesRegex(ValueError, 'not empty'):
            self.run_batch()
        self.factory.assert_not_called()

    def test_simultaneous_batch_is_rejected(self):
        self.output.mkdir()
        with lock(self.output/'.evaluation.lock'):
            with self.assertRaisesRegex(ValueError, 'Another evaluation'):
                self.run_batch()
        self.factory.assert_not_called()

    def test_stale_csv_is_rebuilt_from_checkpoint(self):
        self.run_batch()
        (self.output/'evaluated.csv').write_text('stale')
        self.run_batch()
        self.assertEqual(len(pd.read_csv(self.output/'evaluated.csv')), 4)
        self.assertEqual(self.factory.return_value.evaluate.call_count, 4)

    def test_duplicate_and_invalid_designs_fail_before_cfd(self):
        frame = sample(4)
        frame.loc[1, 'sample_id'] = 0
        frame.to_csv(self.design, index=False)
        with self.assertRaisesRegex(ValueError, 'Duplicate or negative'):
            self.run_batch()
        frame = sample(4)
        frame.loc[1, 'ptl_porosity'] = 1.2
        frame.to_csv(self.design, index=False)
        with self.assertRaisesRegex(ValueError, 'Invalid physical inputs'):
            self.run_batch()
        self.factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
