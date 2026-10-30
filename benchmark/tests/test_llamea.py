"""LLaMEA protocol tests: real framework, scripted LLM replies, mocked CFD.

Numerical-worker/config tests also run in the existing Python 3.9 environment.
The project integration tests run when llamea==1.4.2 is installed on Python 3.11+.
No test makes a network request or invokes an OpenFOAM executable.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import importlib.metadata
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from aemec_opt.engine import ObjectiveResult
from aemec_benchmark.algorithms import ALGORITHMS
from aemec_benchmark.cli import main
from aemec_benchmark.llamea import (
    EXAMPLE, LLaMEAOptimizer, LLaMEASettings, make_llm, protocol,
    require_llamea, run_proposal,
)
from aemec_benchmark.llamea_worker import validate_code
from aemec_benchmark.metrics import hypervolume_2d
from aemec_benchmark.runner import BenchmarkSettings, run_benchmark
from test_benchmark import synthetic_problem
from test_proxy_search import write_pair

try:
    HAS_LLAMEA = sys.version_info >= (3, 11) and importlib.metadata.version('llamea') == '1.4.2'
except importlib.metadata.PackageNotFoundError:
    HAS_LLAMEA = False

A = 'def propose(x, y, attempts, reference, rng):\n    return [0.8, 0.8, 0.8]\n'
B = 'def propose(x, y, attempts, reference, rng):\n    return [0.2, 1.0, 0.5]\n'
INITIAL = np.array([[0., 0., 0.], [.25, .25, .25], [.75, .75, .75], [1., 1., 1.]])


class WorkerAndConfigTests(unittest.TestCase):
    def test_numerical_function_and_linear_algebra_run_in_the_worker(self):
        x = run_proposal(EXAMPLE, [], [], [], [2., 2.], 42, 10.)
        y = run_proposal(EXAMPLE, [], [], [], [2., 2.], 42, 10.)
        np.testing.assert_array_equal(x, y)
        code = ('import numpy as np\ndef propose(x, y, attempts, reference, rng):\n'
                '    return np.linalg.solve(np.eye(3), np.full(3, 0.5))\n')
        np.testing.assert_array_equal(run_proposal(code, [], [], [], [2., 2.], 2, 10), [.5]*3)

    def test_rejects_io_introspection_wrong_interface_and_invalid_points(self):
        for code in (
                'import os\n'+A,
                'import numpy as np\n'+A.replace('return [0.8, 0.8, 0.8]', 'return np.load("file.npy")'),
                A.replace('return [0.8, 0.8, 0.8]', 'return x.__class__'),
                'class Optimizer:\n    pass\n'+A,
                A.replace('x, y, attempts, reference, rng', 'x, y'),
        ):
            with self.subTest(code=code), self.assertRaises(ValueError):
                validate_code(code)
        for point in ('[0., 1.]', '[0., 1., 2.]', '[float("nan"), 0., 0.]'):
            with self.subTest(point=point), self.assertRaisesRegex(ValueError, 'finite values'):
                run_proposal(A.replace('[0.8, 0.8, 0.8]', point), [], [], [], [2., 2.], 1, 10)

    def test_timeout_kills_nonterminating_generated_code(self):
        code = 'def propose(x, y, attempts, reference, rng):\n    while True:\n        pass\n'
        with self.assertRaisesRegex(ValueError, 'exceeded'):
            run_proposal(code, [], [], [], [2., 2.], 1, .3)

    def test_worker_has_no_credentials_or_repository_pythonpath(self):
        import subprocess
        real_run = subprocess.run
        captured = []
        def record(*args, **kwargs):
            captured.append(kwargs['env'])
            return real_run(*args, **kwargs)
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-secret', 'PYTHONPATH': 'private-path'}), \
                patch('aemec_benchmark.llamea.subprocess.run', side_effect=record):
            run_proposal(A, [], [], [], [2., 2.], 1, 10)
        self.assertNotIn('OPENAI_API_KEY', captured[0])
        self.assertNotIn('PYTHONPATH', captured[0])
        self.assertNotIn('test-secret', str(captured[0]))

    def test_configuration_and_effective_budget(self):
        config = LLaMEASettings(model='scripted')
        for invalid in (replace(config, model=''), replace(config, candidates=0),
                        replace(config, candidate_budget=0), replace(config, proposal_timeout=np.inf),
                        replace(config, request_timeout=0), replace(config, provider='missing'),
                        replace(config, base_url='https://user:secret@example.com/v1'),
                        replace(config, api_key_env='not a name')):
            with self.assertRaises(ValueError):
                invalid.validate()
        self.assertEqual(protocol(config, 4, 7)['candidate_budget_effective'], 3)
        self.assertEqual(protocol(config, 4, 7)['candidate_limit_effective'], 1)
        with self.assertRaisesRegex(ValueError, 'requires model'):
            BenchmarkSettings(algorithms=('llamea',)).validate()
        with patch('aemec_benchmark.llamea.sys.version_info', (3, 9)):
            with self.assertRaisesRegex(ValueError, 'Python 3.11'):
                require_llamea(config)

    def test_cli_dry_run_requires_model_but_no_key_package_or_cfd(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'run'
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(['--algorithms', 'llamea', '--output', str(out), '--dry-run']), 2)
            captured = io.StringIO()
            with redirect_stdout(captured), patch('aemec_benchmark.runner.require_llamea') as runtime:
                self.assertEqual(main(['--algorithms', 'llamea', '--llamea-model', 'chosen-model',
                                       '--llamea-provider', 'openai', '--budget', '11', '--initial', '4',
                                       '--seeds', '42', '--output', str(out), '--dry-run']), 0)
                runtime.assert_not_called()
            self.assertIn('11 actual evaluator calls', captured.getvalue())
            self.assertIn('within the existing budget', captured.getvalue())
            self.assertFalse(out.exists())


@unittest.skipUnless(HAS_LLAMEA, 'Install requirements-llamea.txt on Python 3.11+ for native API tests')
class NativeLLaMEATests(unittest.TestCase):
    def setUp(self):
        self.module = require_llamea()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.config = LLaMEASettings(model='scripted', candidates=2, candidate_budget=2)

    def scripted(self, codes):
        module = self.module
        class Scripted(module.LLM):
            def __init__(self):
                super().__init__(api_key='', model='scripted')
                self.responses = list(codes)
                self.prompts = []
            def query(self, session):
                self.prompts.append(session)
                return '# Description: scripted test optimizer\n```python\n'+self.responses.pop(0)+'```'
        return Scripted()

    def optimizer(self, codes=(A, B), budget=11, failed=()):
        llm = self.scripted(codes)
        patcher = patch('aemec_benchmark.llamea.make_llm', return_value=llm)
        patcher.start()
        self.addCleanup(patcher.stop)
        alg = LLaMEAOptimizer(42, [2., 2.], 4, budget, self.config, self.folder)
        for i, fixed in enumerate(INITIAL):
            x, _ = alg.ask(fixed)
            np.testing.assert_array_equal(x, fixed)
            alg.tell(x, None if i in failed else [1.5, 1.5])
        self.assertEqual(llm.prompts, [])
        return alg, llm

    def test_native_generation_mutation_selection_and_winner_deployment(self):
        alg, llm = self.optimizer()
        snapshots = []
        real = run_proposal
        def capture(code, x, y, attempts, reference, seed, timeout):
            snapshots.append((code, len(x), len(attempts)))
            return real(code, x, y, attempts, reference, seed, timeout)
        phases = []
        with patch('aemec_benchmark.llamea.run_proposal', side_effect=capture):
            for _ in range(7):
                point, details = alg.ask()
                phases.append(details['proposal'])
                alg.tell(point, point[:2])
        self.assertEqual(phases, ['llamea_candidate']*4 + ['llamea_deployment']*3)
        self.assertEqual(len(llm.prompts), 2)
        self.assertEqual(snapshots[0][1:], (4, 4))
        self.assertEqual(snapshots[2][1:], (4, 4))  # Candidate B cannot see A's CFD data.
        self.assertEqual(snapshots[4][1:], (8, 8))  # Winner sees this run's entire archive.
        self.assertIsInstance(alg.engine, self.module.LLaMEA)
        self.assertEqual(alg.engine.population[0].code, B.rstrip())
        self.assertEqual((alg.folder/'best_optimizer.py').read_text(), B.rstrip())
        records = json.loads((alg.folder/'candidates.json').read_text())
        self.assertAlmostEqual(records[0]['fitness'], hypervolume_2d([[1.5, 1.5], [.8, .8]], [2., 2.]))
        self.assertAlmostEqual(records[1]['fitness'], hypervolume_2d([[1.5, 1.5], [.2, 1.]], [2., 2.]))
        with self.assertRaisesRegex(ValueError, 'budget is exhausted'):
            alg.ask()

    def test_failures_charge_candidate_steps_without_inventing_objectives(self):
        alg, _ = self.optimizer(failed=(0, 1, 2, 3))
        self.assertEqual(alg.shared_y, [])
        for _ in range(2):
            x, _ = alg.ask()
            alg.tell(x, None)
        self.assertEqual(len(alg.attempts), 6)
        self.assertEqual(alg.records[0]['steps'], 2)
        self.assertEqual(alg.records[0]['fitness'], 0.)
        self.assertEqual(alg.records[0]['successful_new'], 0)
        self.assertEqual(alg.y, [])

    def test_invalid_candidates_are_bounded_and_never_silently_fallback(self):
        bad = 'def wrong_interface():\n    return 1\n'
        alg, llm = self.optimizer(codes=(bad, bad))
        with self.assertRaisesRegex(ValueError, 'no valid optimizer'):
            alg.ask()
        self.assertEqual(len(llm.prompts), 2)
        self.assertEqual(len(alg.attempts), 4)
        self.assertEqual(len(alg.records), 2)
        self.assertTrue(all(r['fitness'] == -1 for r in alg.records))
        self.assertIsNone(alg.best)
        self.assertEqual(alg.y, [[1.5, 1.5]]*4)

    def test_one_rejected_candidate_can_be_repaired_and_remaining_budget_is_used(self):
        alg, _ = self.optimizer(codes=('def wrong():\n    pass\n', B))
        for _ in range(7):
            x, _ = alg.ask()
            alg.tell(x, x[:2])
        self.assertEqual(len(alg.attempts), 11)
        self.assertEqual(alg.records[0]['steps'], 0)
        self.assertEqual(alg.records[1]['steps'], 2)
        self.assertIsNotNone(alg.best)

    def test_global_rng_and_pending_protocol_are_preserved(self):
        alg, _ = self.optimizer()
        old_np, old_py = np.random.get_state(), random.getstate()
        x, _ = alg.ask()
        after = np.random.get_state()
        np.testing.assert_array_equal(old_np[1], after[1])
        self.assertEqual(old_np[2:], after[2:])
        self.assertEqual(old_py, random.getstate())
        with self.assertRaises(ValueError):
            alg.ask()
        with self.assertRaises(ValueError):
            alg.tell([.1, .1, .1], [1., 1.])
        with self.assertRaises(ValueError):
            alg.tell(x, [np.nan, 0.])
        alg.tell(x, None)

    def test_actual_cli_uses_only_cfd_and_exact_shared_attempt_count(self):
        llm = self.scripted([A, B])
        out = self.folder/'benchmark'
        proxies = write_pair(self.folder/'models')
        with patch('aemec_benchmark.llamea.make_llm', return_value=llm), \
                patch('aemec_benchmark.evaluation.shutil.which', return_value='/mock/tool'), \
                patch('aemec_opt.case.AemecOpenFoamEvaluator') as adapter:
            adapter.return_value.evaluate.return_value = SimpleNamespace(
                as_objective_result=lambda: ObjectiveResult((2., 3e-9), {}))
            self.assertEqual(main(['--algorithms', *ALGORITHMS, '--llamea-model', 'scripted',
                                   '--proxy-files', *map(str, proxies), '--proxy-candidates', '16',
                                   '--gp-candidates', '16', '--gp-restarts', '0',
                                   '--llamea-candidates', '2', '--llamea-candidate-budget', '2',
                                   '--budget', '11', '--initial', '4', '--seeds', '42', '--output', str(out)]), 0)
            self.assertEqual(adapter.return_value.evaluate.call_count, 4+6*7)
        manifest = json.loads((out/'manifest.json').read_text())
        self.assertEqual(manifest['actual_evaluator_calls'], 46)
        self.assertEqual(manifest['versions']['llamea'], '1.4.2')
        self.assertEqual(manifest['llamea_protocol']['candidate_limit_effective'], 2)
        for algorithm in ALGORITHMS:
            df = pd.read_csv(out/f'seed_42/{algorithm}/evaluations.csv')
            self.assertEqual(len(df), 11)
            self.assertTrue(np.allclose(df.cell_voltage_v, 2.))
            self.assertTrue(np.allclose(df.crossover_rate_mol_s, 3e-9, rtol=1e-12, atol=0))
        self.assertEqual(len(llm.prompts), 2)

    def test_missing_key_rejected_before_output_or_cfd(self):
        config = replace(self.config, provider='openai', api_key_env='AEMEC_TEST_MISSING_KEY')
        out = self.folder/'run'
        with patch.dict(os.environ, {}, clear=True), \
                patch('aemec_opt.case.AemecOpenFoamEvaluator') as evaluator:
            with self.assertRaisesRegex(ValueError, 'AEMEC_TEST_MISSING_KEY'):
                run_benchmark(synthetic_problem(), BenchmarkSettings(algorithms=('llamea',), llamea=config), out)
            evaluator.assert_not_called()
        self.assertFalse(out.exists())

    def test_provider_audit_omits_keys_and_redacts_service_errors(self):
        config = replace(self.config, provider='openai')
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'secret-test-value'}), patch('openai.OpenAI') as provider:
            client = provider.return_value
            client.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='reply'))], usage=None)
            llm = make_llm(self.module, config, self.folder)
            self.assertEqual(llm.query([{'role': 'user', 'content': 'task'}]), 'reply')
            client.chat.completions.create.side_effect = RuntimeError('secret-test-value')
            with self.assertRaisesRegex(RuntimeError, 'request failed'):
                llm.query([{'role': 'user', 'content': 'task'}])
        audit = ''.join(p.read_text() for p in self.folder.glob('llm_*.json'))
        self.assertNotIn('secret-test-value', audit)
        self.assertIn('RuntimeError', audit)
        self.assertEqual(provider.call_args.kwargs['max_retries'], 0)


if __name__ == '__main__':
    unittest.main()
