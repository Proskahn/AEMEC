from __future__ import annotations

import unittest
import csv
import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aemec_opt.cli import build_parser, run_optimization, validate_args
from aemec_opt.engine import ObjectiveResult
from run_optimization import OptimizationError, main


class CliTests(unittest.TestCase):
    def test_default_cli_keeps_legacy_options_and_fifty_evaluations(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.iterations, 50)
        self.assertEqual(args.run_mode, "fast")
        self.assertEqual(args.solver_iterations, 250)
        self.assertEqual(args.stability_samples, 5)
        self.assertEqual(args.min_water_inlet_temperature_k, 298.15)
        self.assertEqual(args.max_water_inlet_temperature_k, 353.15)
        config, _ = validate_args(args)
        self.assertEqual([p.name for p in config.parameters], ["membrane_thickness_um", "water_inlet_temperature_k"])

    def test_temperature_bounds_are_configurable_and_validated(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--min-water-inlet-temperature-k", "313.15", "--max-water-inlet-temperature-k", "343.15"])
        config, _ = validate_args(args)
        self.assertEqual((config.parameters[1].lower_bound, config.parameters[1].upper_bound), (313.15, 343.15))
        for lower, upper in (("nan", "353"), ("0", "353"), ("353", "313"), ("313", "313")):
            with self.subTest(lower=lower, upper=upper), self.assertRaises(OptimizationError):
                validate_args(parser.parse_args(["--min-water-inlet-temperature-k", lower, "--max-water-inlet-temperature-k", upper]))

    def test_compatibility_entrypoint_exports_shared_error(self) -> None:
        self.assertTrue(issubclass(OptimizationError, RuntimeError))
        self.assertTrue(callable(main))

    def test_joint_study_passes_both_parameters_to_adapter_and_reports_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "template").write_text("unchanged")
            args = build_parser().parse_args([
                "--case", str(source), "--work-dir", str(root / "work"),
                "--output-dir", str(root / "results"), "--iterations", "3", "--startup-trials", "3",
            ])

            def evaluate(number, thickness, temperature):
                objective = SimpleNamespace(cell_voltage_v=2 + thickness / 1000 - temperature / 10000, crossover_rate_mol_s=temperature / thickness * 1e-8, target_current_density_a_m2=10000)
                return SimpleNamespace(
                    objective=objective, polarization_curve_csv="curve.csv", polarization_curve_plot="curve.png",
                    as_objective_result=lambda: ObjectiveResult((objective.cell_voltage_v, objective.crossover_rate_mol_s), {}),
                )

            with patch("aemec_opt.cli.AemecOpenFoamEvaluator") as adapter, redirect_stdout(io.StringIO()):
                adapter.return_value.evaluate.side_effect = evaluate
                study = run_optimization(args)
                self.assertEqual(adapter.return_value.evaluate.call_count, 3)
                for trial, call in zip(study.trials, adapter.return_value.evaluate.call_args_list):
                    self.assertEqual(call.args, (trial.number, trial.params["membrane_thickness_um"], trial.params["water_inlet_temperature_k"]))
                run_optimization(args)
                self.assertEqual(adapter.return_value.evaluate.call_count, 3)
            for filename in ("optimization_results.csv", "pareto_front.csv", "knee_points.csv"):
                path = next((root / "results").rglob(filename))
                with path.open() as handle:
                    for row in csv.DictReader(handle):
                        trial = study.trials[int(row["trial"])]
                        self.assertEqual(float(row["water_inlet_temperature_k"]), trial.params["water_inlet_temperature_k"])
            self.assertEqual((source / "template").read_text(), "unchanged")


if __name__ == "__main__":
    unittest.main()
