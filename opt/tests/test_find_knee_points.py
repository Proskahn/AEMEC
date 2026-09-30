from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.find_knee_points import main


class FindKneePointsCommandTests(unittest.TestCase):
    def test_command_writes_knee_csv_and_annotated_plot(self) -> None:
        points = (
            (0.0, 10.0),
            (2.0, 9.0),
            (4.0, 6.0),
            (5.5, 5.5),
            (6.5, 2.0),
            (10.0, 0.0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            study_dir = Path(tmp)
            with (study_dir / "optimization_results.csv").open(
                "w", newline="", encoding="utf-8"
            ) as output_file:
                writer = csv.writer(output_file)
                writer.writerow(
                    [
                        "trial",
                        "membrane_thickness_um",
                        "cell_voltage_v",
                        "crossover_rate_mol_s",
                    ]
                )
                for trial, objectives in enumerate(points):
                    writer.writerow([trial, 10.0 + trial, *objectives])

            with redirect_stdout(io.StringIO()):
                result = main([str(study_dir)])

            self.assertEqual(result, 0)
            self.assertTrue((study_dir / "pareto_front.png").is_file())
            with (study_dir / "knee_points.csv").open(
                newline="", encoding="utf-8"
            ) as input_file:
                rows = list(csv.DictReader(input_file))
            self.assertEqual(
                {row["method"]: row["trial"] for row in rows},
                {"chebyshev": "3", "bend_angle": "4"},
            )

    def test_joint_study_knees_retain_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            study_dir = Path(tmp)
            with (study_dir / "optimization_results.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["trial", "membrane_thickness_um", "water_inlet_temperature_k", "cell_voltage_v", "crossover_rate_mol_s"])
                writer.writerows([(0, 20, 343.15, 1, 4), (1, 40, 323.15, 2, 2), (2, 60, 303.15, 4, 1)])
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(study_dir)]), 0)
            with (study_dir / "knee_points.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertTrue(all(float(row["water_inlet_temperature_k"]) == 323.15 for row in rows))

    def test_three_variable_study_knees_retain_porosity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            study_dir = Path(tmp)
            with (study_dir / "optimization_results.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["trial", "membrane_thickness_um", "water_inlet_temperature_k", "ptl_porosity", "cell_voltage_v", "crossover_rate_mol_s"])
                writer.writerows([(0, 20, 343.15, 0.4, 1, 4), (1, 40, 323.15, 0.6, 2, 2), (2, 60, 303.15, 0.8, 4, 1)])
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main([str(study_dir)]), 0)
            self.assertIn("ptl_porosity=0.6", output.getvalue())
            with (study_dir / "knee_points.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertTrue(all(float(row["ptl_porosity"]) == 0.6 for row in rows))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
