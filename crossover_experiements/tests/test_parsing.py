from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crossover_experiments.parsing import (
    FARADAY_CONSTANT_C_MOL,
    parse_experiment_log,
    read_timeseries_csv,
    summarize_samples,
    write_timeseries_csv,
)


def fake_log() -> str:
    records = []
    for index, (time_s, current_density, voltage, crossover, accepted) in enumerate(
        (
            (15.0, -9900.0, 1.89, 5.0e-8, "false"),
            (19.9, -10000.0, 1.90, 4.5e-8, "false"),
            (20.0, -10000.0, 1.90, 4.0e-8, "true"),
        )
    ):
        current_a = current_density * 8.0e-5
        relative_error = abs(current_density + 10000.0) / 10000.0
        records.extend(
            (
                f"Time = {time_s}",
                (
                    "Controlled boundary current (A) at interconnect0: "
                    f"signed = {current_a}, magnitude = {abs(current_a)}, "
                    f"current density = {current_density} A/m2, voltage = {voltage}"
                ),
                f"Hydrogen crossover objective: anode gas source rate = {crossover} mol/s",
                (
                    "galvanostatic target: -10000 A/m2, requested target: 10000 A/m2, "
                    f"signed target: -10000 A/m2, measured current density: {current_density} A/m2, "
                    f"current error: 0 A/m2, current relative error: {relative_error}, "
                    f"voltage: {voltage}, stability samples: {min(index + 3, 5)}/5, "
                    f"accepted: {accepted}, polarizationComplete: false"
                ),
            )
        )
    records.append("End")
    return "\n".join(records)


class ExperimentParsingTests(unittest.TestCase):
    def test_parse_derived_values_and_controller_state(self) -> None:
        samples = parse_experiment_log(fake_log())
        self.assertEqual(len(samples), 3)
        final = samples[-1]
        self.assertAlmostEqual(final.current_density_magnitude_a_cm2, 1.0)
        self.assertAlmostEqual(
            final.hydrogen_production_rate_mol_s,
            0.8 / (2.0 * FARADAY_CONSTANT_C_MOL),
        )
        self.assertTrue(final.accepted)
        self.assertEqual(final.stable_samples, 5)

    def test_csv_round_trip_and_final_window_summary(self) -> None:
        samples = parse_experiment_log(fake_log())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "timeseries.csv"
            write_timeseries_csv(path, samples)
            restored = read_timeseries_csv(path)
        self.assertEqual(restored, samples)
        summary = summarize_samples(40.0, restored, final_window_s=5.0)
        self.assertEqual(summary.window_sample_count, 3)
        self.assertAlmostEqual(summary.mean_voltage_v, (1.89 + 1.9 + 1.9) / 3.0)
        self.assertTrue(summary.final_controller_accepted)


if __name__ == "__main__":
    unittest.main()
