from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.plot_membrane_polarization_curves import (
    load_selected_curves,
    plot_curves,
)


class MembranePolarizationCurvePlotTests(unittest.TestCase):
    def test_loads_requested_trials_and_writes_combined_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            study_dir = Path(directory)
            logs_dir = study_dir / "logs"
            logs_dir.mkdir()
            with (study_dir / "optimization_results.csv").open(
                "w", newline="", encoding="utf-8"
            ) as output_file:
                writer = csv.writer(output_file)
                writer.writerow(["trial", "membrane_thickness_um"])
                for trial, thickness in enumerate((20, 40, 60, 80)):
                    writer.writerow([trial, thickness])

            for trial, thickness in enumerate((20, 40, 60, 80)):
                curve_path = (
                    logs_dir / f"trial_{trial:04d}_polarization_curve.csv"
                )
                with curve_path.open("w", newline="", encoding="utf-8") as output_file:
                    writer = csv.writer(output_file)
                    writer.writerow(
                        [
                            "cell_voltage_v",
                            "final_current_density_magnitude_a_cm2",
                            "final_crossover_rate_mol_s",
                        ]
                    )
                    writer.writerow(
                        [1.6 + thickness / 1000.0, 0.4, 8.0e-8 / thickness]
                    )
                    writer.writerow(
                        [1.8 + thickness / 1000.0, 1.0, 1.0e-7 / thickness]
                    )

            curves = load_selected_curves(study_dir)
            output_path = (
                study_dir / "selected_polarization_and_crossover_curves.png"
            )
            plot_curves(curves, output_path)

            self.assertEqual(
                [curve.membrane_thickness_um for curve in curves],
                [20.0, 40.0, 60.0, 80.0],
            )
            self.assertEqual(curves[0].crossover_rate_mol_s, (4.0e-9, 5.0e-9))
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_reports_an_unavailable_requested_thickness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            study_dir = Path(directory)
            with (study_dir / "optimization_results.csv").open(
                "w", newline="", encoding="utf-8"
            ) as output_file:
                writer = csv.writer(output_file)
                writer.writerow(["trial", "membrane_thickness_um"])
                writer.writerow([0, 20])

            with self.assertRaisesRegex(ValueError, "no completed trial found at 40"):
                load_selected_curves(study_dir, (40.0,))


if __name__ == "__main__":
    unittest.main()
