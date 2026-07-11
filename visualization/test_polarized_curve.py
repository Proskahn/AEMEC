from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from visualization.polarized_curve import (
    Sample,
    last_sample_per_target_current,
    parse_log,
    split_target_convergence,
    write_csv,
)


class PolarizationCurveTests(unittest.TestCase):
    def test_boundary_current_density_and_galvanostatic_targets_are_extracted(self) -> None:
        log = """
Time = 1
galvanostatic target: -10000 A/m2, raw dV: 0.1, limited dV: 0.005
ibar: -9000 voltage: 1.7
Controlled boundary current (A) at x: signed = -0.7, magnitude = 0.7, current density = -8750 A/m2, voltage = 1.75
Time = 2
galvanostatic target: -10000 A/m2, raw dV: 0.01, limited dV: 0.001
Controlled boundary current (A) at x: signed = -0.8, magnitude = 0.8, current density = -10000 A/m2, voltage = 1.8
Time = 3
galvanostatic target: -20000 A/m2, raw dV: 0.01, limited dV: 0.001
Controlled boundary current (A) at x: signed = -1.6, magnitude = 1.6, current density = -20000 A/m2, voltage = 1.9
"""

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "log.run"
            log_path.write_text(log, encoding="utf-8")
            samples = parse_log(log_path, active_area_cm2=999.0)

            self.assertEqual(len(samples), 3)
            self.assertEqual(samples[0].source, "boundary-current-density")
            self.assertEqual(samples[0].current_density_a_m2, -8750.0)
            curve = last_sample_per_target_current(samples, current_precision=3)
            self.assertEqual(len(curve), 2)
            self.assertEqual(curve[0].voltage_v, 1.8)
            self.assertEqual(curve[1].voltage_v, 1.9)

            accepted, rejected = split_target_convergence(
                samples,
                relative_tolerance=0.05,
                zero_target_absolute_tolerance_a_m2=100.0,
            )
            self.assertEqual(len(accepted), 2)
            self.assertEqual(len(rejected), 1)

            csv_path = Path(directory) / "curve.csv"
            write_csv(curve, csv_path)
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("target_current_density_a_m2", header)

    def test_ibar_is_used_only_when_boundary_current_is_absent(self) -> None:
        log = """
Time = 1
galvanostatic target: -10000 A/m2, raw dV: 0, limited dV: 0
ibar: -9000 voltage: 1.7
"""

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "log.run"
            log_path.write_text(log, encoding="utf-8")
            samples = parse_log(log_path, active_area_cm2=0.8)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].source, "ibar-fallback")
        self.assertEqual(samples[0].current_density_a_m2, -9000.0)

    def test_zero_and_unsettled_targets_are_rejected(self) -> None:
        samples = [
            Sample(10.0, 1.0, -0.8, -10000.0, 0.0, "boundary-current-density"),
            Sample(40.0, 1.020577, -0.4584026, -5730.033, -6000.0, "boundary-current-density"),
            Sample(60.0, 1.988741, -0.6451845, -8064.807, -10000.0, "boundary-current-density"),
        ]

        accepted, rejected = split_target_convergence(
            samples,
            relative_tolerance=0.05,
            zero_target_absolute_tolerance_a_m2=100.0,
        )

        self.assertEqual([sample.time for sample in accepted], [40.0])
        self.assertEqual([sample.time for sample in rejected], [10.0, 60.0])


if __name__ == "__main__":
    unittest.main()
