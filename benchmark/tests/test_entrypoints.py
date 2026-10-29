"""Exercise relocated launchers in fresh interpreters outside the checkout.

In-process imports can conceal an incorrect sys.path or repository-root lookup.
These checks run the real command with no inherited PYTHONPATH and no CFD calls.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class EntrypointTests(unittest.TestCase):
    def run_script(self, script, directory, *args):
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        env['MPLCONFIGDIR'] = str(directory/'matplotlib')
        result = subprocess.run([sys.executable, str(ROOT/script), *map(str, args)],
                                cwd=directory, env=env, capture_output=True, text=True,
                                timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
        return result.stdout

    def test_benchmark_dry_run_resolves_shared_adapter_and_new_config(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            output = folder/'results'
            stdout = self.run_script('benchmark/run_benchmark.py', folder,
                                     '--output', output, '--dry-run', '--budget', '8',
                                     '--initial', '4', '--seeds', '42')
            self.assertIn('Evaluator: CFD (AEMEC/OpenFOAM)', stdout)
            self.assertIn('20 actual evaluator calls', stdout)
            self.assertIn(str(ROOT/'benchmark/configs/aemec_benchmark.json'), stdout)
            self.assertIn(str(ROOT/'run/AEMEC'), stdout)
            self.assertFalse(output.exists())

    def test_optimization_launcher_still_runs_independently(self):
        with tempfile.TemporaryDirectory() as temp:
            stdout = self.run_script('opt/run_optimization.py', Path(temp), '--help')
            self.assertIn('--iterations', stdout)
            self.assertIn('--study-name', stdout)

    def test_legacy_benchmark_launcher_forwards_to_cfd_only_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            stdout = self.run_script('opt/run_benchmark.py', Path(temp), '--help')
            self.assertIn('ela_proxy', stdout)
            self.assertIn('--backend {aemec}', stdout)
            self.assertNotIn('zdt1', stdout)


if __name__ == '__main__':
    unittest.main()
