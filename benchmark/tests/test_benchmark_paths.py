"""Ensure relocating proxy artifacts does not silently change the benchmark."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from aemec_benchmark.runner import INPUTS
from aemec_benchmark.cli import ROOT, build_parser, main
from aemec_benchmark.proxy_assets import load_proxy_assets, resolve_proxy_paths


def write_proxy(path, landscape):
    """Make a minimal real artifact so path tests also exercise schema loading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'schema_version': 1, 'landscape': landscape, 'tree': {'op': 'x'},
        'input_columns': INPUTS, 'input_lower': [10., 298.15, .4],
        'input_upper': [100., 353.15, .8], 'raw_offset': 0., 'raw_scale': 1.}))
    return path


def write_pair(root, rank=1):
    return [write_proxy(root/name/f'proxy_{rank:02d}'/'proxy.json', name)
            for name in ('cell_voltage_v', 'crossover_rate_mol_s')]


class ProxyPathTests(unittest.TestCase):
    def args(self, *options):
        return build_parser().parse_args(['--algorithms', 'ela_proxy', '--output', 'unused', *map(str, options)])

    def test_discovery_in_a_renamed_nested_directory_respects_rank(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            write_pair(root/'ELA/results/rank_one_only')
            expected = write_pair(root/'ELA/results/new location/run_7', rank=2)
            with patch('aemec_benchmark.proxy_assets.ROOT', root):
                args = self.args('--proxy-rank', '2')
                self.assertEqual(resolve_proxy_paths(args), expected)
                assets = load_proxy_assets(resolve_proxy_paths(args), [10., 298.15, .4], [100., 353.15, .8])
            np.testing.assert_allclose([model(assets.lower) for model in assets.models], [0, 0])
            self.assertEqual([model['path'] for model in assets.metadata['models']], list(map(str, expected)))

    def test_ambiguous_discovery_requires_choice_and_explicit_root_wins(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            a, b = root/'ELA/results/first run', root/'ELA/results/second'
            expected = write_pair(a)
            write_pair(b)
            with patch('aemec_benchmark.proxy_assets.ROOT', root):
                with self.assertRaisesRegex(ValueError, 'Multiple rank-1 proxy pairs') as caught:
                    resolve_proxy_paths(self.args())
                self.assertIn(str(a), str(caught.exception))
                self.assertIn(str(b), str(caught.exception))
                self.assertEqual(resolve_proxy_paths(self.args('--proxy-root', a)), expected)

    def test_never_combines_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_proxy(root/'ELA/results/a/cell_voltage_v/proxy_01/proxy.json', 'cell_voltage_v')
            write_proxy(root/'ELA/results/b/crossover_rate_mol_s/proxy_01/proxy.json', 'crossover_rate_mol_s')
            with patch('aemec_benchmark.proxy_assets.ROOT', root):
                with self.assertRaisesRegex(ValueError, 'No complete rank-1 proxy pair') as caught:
                    resolve_proxy_paths(self.args())
            self.assertIn('copy the generated proxies', str(caught.exception))

    def test_explicit_missing_root_does_not_substitute_discovered_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_pair(root/'ELA/results/available')
            with patch('aemec_benchmark.proxy_assets.ROOT', root):
                with self.assertRaisesRegex(ValueError, r'Proxy JSON file\(s\) not found') as caught:
                    resolve_proxy_paths(self.args('--proxy-root', root/'wrong'))
            self.assertIn('wrong/cell_voltage_v/proxy_01/proxy.json', str(caught.exception))

    def test_explicit_files_support_arbitrary_names_and_check_roles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            voltage = write_proxy(root/'voltage renamed.json', 'cell_voltage_v')
            crossover = write_proxy(root/'another folder/crossover.json', 'crossover_rate_mol_s')
            # Explicit relative paths resolve against the calling shell, even
            # when they do not live under this repository's ELA/results folder.
            args = self.args('--proxy-files', os.path.relpath(voltage), os.path.relpath(crossover))
            self.assertEqual(resolve_proxy_paths(args), [voltage, crossover])
            load_proxy_assets(resolve_proxy_paths(args), [10., 298.15, .4], [100., 353.15, .8])
            with self.assertRaisesRegex(ValueError, 'Proxy objective roles'):
                load_proxy_assets(resolve_proxy_paths(self.args('--proxy-files', crossover, voltage)), [10., 298.15, .4], [100., 353.15, .8])

    def test_dry_run_prints_discovered_files_without_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            paths = write_pair(root/'ELA/results/server_run')
            out = root/'output'
            config = root/'benchmark/configs/aemec_benchmark.json'
            config.parent.mkdir(parents=True)
            shutil.copyfile(ROOT/'benchmark/configs/aemec_benchmark.json', config)
            (root/'run/AEMEC').mkdir(parents=True)
            stdout = io.StringIO()
            with patch('aemec_benchmark.cli.ROOT', root), patch('aemec_benchmark.proxy_assets.ROOT', root), redirect_stdout(stdout):
                self.assertEqual(main(['--algorithms', 'ela_proxy', '--output', str(out), '--dry-run']), 0)
            self.assertFalse(out.exists())
            for path in paths:
                self.assertIn(str(path), stdout.getvalue())

    def test_missing_results_directory_and_invalid_rank_are_actionable(self):
        with tempfile.TemporaryDirectory() as temp, patch('aemec_benchmark.proxy_assets.ROOT', Path(temp)):
            with self.assertRaisesRegex(ValueError, 'No complete rank-1 proxy pair'):
                resolve_proxy_paths(self.args())
            with self.assertRaisesRegex(ValueError, 'proxy-rank must be positive'):
                resolve_proxy_paths(self.args('--proxy-rank', '0'))


if __name__ == '__main__':
    unittest.main()
