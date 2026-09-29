from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from crossover_experiments.current_sweep_cli import main


class ThicknessSweepCliTests(unittest.TestCase):
    def test_four_independent_sweeps_and_overlay(self):
        with patch('crossover_experiments.current_sweep_cli.CurrentSweepRunner') as runner, \
             patch('crossover_experiments.current_sweep_cli.read_current_sweep_timeseries', return_value=[]) , \
             patch('crossover_experiments.current_sweep_cli.summarize_current_sweep', return_value=['point']), \
             patch('crossover_experiments.current_sweep_cli.plot_thickness_current_sweeps') as plot:
            runner.return_value.run.return_value = 0
            self.assertEqual(main(['--thicknesses', '20', '40', '60', '80',
                                   '--output-dir', '/tmp/results', '--work-dir', '/tmp/work']), 0)
            self.assertEqual(runner.call_count, 4)
            for call, thickness in zip(runner.call_args_list, (20, 40, 60, 80)):
                self.assertEqual(call.kwargs['config'].membrane_thickness_um, thickness)
                self.assertEqual(call.kwargs['output_dir'], Path(f'/tmp/results/{thickness}um'))
                self.assertEqual(call.kwargs['work_dir'], Path(f'/tmp/work/{thickness}um'))
            self.assertEqual([t for t, _ in plot.call_args.args[1]], [20, 40, 60, 80])

    def test_invalid_thickness_rejected_before_any_run(self):
        with patch('crossover_experiments.current_sweep_cli.CurrentSweepRunner') as runner:
            self.assertEqual(main(['--thicknesses', '20', '-40']), 2)
            runner.assert_not_called()

    def test_dry_run_does_not_plot(self):
        with patch('crossover_experiments.current_sweep_cli.CurrentSweepRunner') as runner, \
             patch('crossover_experiments.current_sweep_cli.plot_thickness_current_sweeps') as plot:
            runner.return_value.run.return_value = 0
            self.assertEqual(main(['--thicknesses', '20', '40', '--dry-run']), 0)
            plot.assert_not_called()
