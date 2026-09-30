from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.plot_membrane_polarization_curves import (
    design_label,
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
                        ]
                    )
                    writer.writerow([1.6 + thickness / 1000.0, 0.4])
                    writer.writerow([1.8 + thickness / 1000.0, 1.0])

            curves = load_selected_curves(study_dir)
            output_path = study_dir / "selected_polarization_curves.png"
            plot_curves(curves, output_path)

            self.assertEqual(
                [curve.membrane_thickness_um for curve in curves],
                [20.0, 40.0, 60.0, 80.0],
            )
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

    def test_same_thickness_keeps_both_temperatures_and_trial_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            study_dir = Path(directory)
            (study_dir / "logs").mkdir()
            with (study_dir / "optimization_results.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["trial", "membrane_thickness_um", "water_inlet_temperature_k"])
                writer.writerows([(0, 40, 313.15), (1, 40, 343.15)])
            for trial in (0, 1):
                (study_dir / f"logs/trial_{trial:04d}_polarization_curve.csv").write_text("cell_voltage_v,final_current_density_magnitude_a_cm2\n1.8,1\n")
            curves = load_selected_curves(study_dir, (40,))
            self.assertEqual([c.water_inlet_temperature_k for c in curves], [313.15, 343.15])
            selected = load_selected_curves(study_dir, trial_numbers=[1])
            self.assertEqual([c.trial_number for c in selected], [1])
            plot_curves(curves, study_dir / "comparison.png")

    def test_porosity_and_side_are_retained_in_curve_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            study_dir = Path(directory)
            (study_dir / "logs").mkdir()
            (study_dir / "optimization_results.csv").write_text(
                "trial,membrane_thickness_um,water_inlet_temperature_k,ptl_porosity,ptl_side\n"
                "0,40,333.15,0.4,anode\n1,40,333.15,0.8,anode\n"
            )
            for trial in (0, 1):
                (study_dir / f"logs/trial_{trial:04d}_polarization_curve.csv").write_text("cell_voltage_v,final_current_density_magnitude_a_cm2\n1.8,1\n")
            curves = load_selected_curves(study_dir, trial_numbers=[0, 1])
            self.assertEqual([c.ptl_porosity for c in curves], [0.4, 0.8])
            self.assertIn("anode PTL ε=0.4", design_label(curves[0]))
            self.assertIn("PTL ε=0.8", design_label(curves[1]))
            plot_curves(curves, study_dir / "comparison.png")


if __name__ == "__main__":
    unittest.main()
