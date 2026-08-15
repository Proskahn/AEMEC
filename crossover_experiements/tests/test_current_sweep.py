from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crossover_experiments.config import (
    CurrentSweepConfig,
    configure_current_sweep_case,
)
from crossover_experiments.current_sweep import (
    read_current_sweep_timeseries,
    summarize_current_sweep,
    write_current_sweep_points,
    write_current_sweep_timeseries,
)
from crossover_experiments.current_sweep_reporting import (
    write_current_sweep_reports,
)
from crossover_experiments.parsing import ExperimentSample
from crossover_experiments.project import DEFAULT_SOURCE_CASE, copy_clean_case


def sweep_samples(config: CurrentSweepConfig) -> list[ExperimentSample]:
    samples: list[ExperimentSample] = []
    for target_index, (target, signed_target) in enumerate(
        zip(config.current_targets_a_cm2, config.signed_targets_a_m2)
    ):
        for sample_index in range(5):
            time_s = target_index * config.minimum_hold_s + 19.6 + 0.1 * sample_index
            crossover_rate = (1.0 + target) * 1.0e-8
            samples.append(
                ExperimentSample(
                    time_s=time_s,
                    current_a=signed_target * config.membrane_area_m2,
                    current_density_a_m2=signed_target,
                    voltage_v=1.3 + 0.3 * target,
                    crossover_rate_mol_s=crossover_rate,
                    target_current_density_a_m2=signed_target,
                    current_relative_error=0.0,
                    stable_samples=sample_index + 1,
                    required_stable_samples=5,
                    accepted=sample_index == 4,
                )
            )
    return samples


class CurrentSweepTests(unittest.TestCase):
    def test_case_contains_all_eleven_current_targets(self) -> None:
        config = CurrentSweepConfig()
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary) / "case"
            copy_clean_case(DEFAULT_SOURCE_CASE, case)
            configure_current_sweep_case(case, config)
            controller = (case / "constant/phiEAnode/regionProperties").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "targets                     (0 -2000 -4000 -6000 -8000 -10000 -12000 -14000 -16000 -18000 -20000);",
                controller,
            )
            control = (case / "system/controlDict.run").read_text(encoding="utf-8")
            self.assertRegex(control, r"(?m)^endTime\s+400\s*;")
            initial = (case / "0.orig/phiEAnode/phi").read_text(encoding="utf-8")
            self.assertIn("internalField   uniform 1.3;", initial)
            self.assertIn("zero-current sweep start", initial)

    def test_flux_density_uses_membrane_area_and_zero_fraction_is_undefined(self) -> None:
        config = CurrentSweepConfig()
        samples = sweep_samples(config)
        points = summarize_current_sweep(samples, config)
        self.assertEqual(len(points), 11)
        self.assertAlmostEqual(
            points[0].mean_crossover_flux_density_mol_m2_s,
            1.0e-8 / 8.0e-5,
        )
        self.assertIsNone(points[0].crossover_fraction_percent)
        self.assertGreater(points[-1].hydrogen_production_flux_density_mol_m2_s, 0)

    def test_summary_uses_controller_stability_samples_not_transient(self) -> None:
        config = CurrentSweepConfig()
        samples = sweep_samples(config)
        target = config.current_targets_a_cm2[1]
        signed_target = config.signed_targets_a_m2[1]
        first_stable_time = min(
            sample.time_s
            for sample in samples
            if sample.target_current_density_a_m2 == signed_target
        )
        for sample_index in range(5):
            measured = signed_target * 0.97
            samples.append(
                ExperimentSample(
                    time_s=first_stable_time - 0.5 + 0.1 * sample_index,
                    current_a=measured * config.membrane_area_m2,
                    current_density_a_m2=measured,
                    voltage_v=1.3 + 0.3 * target,
                    crossover_rate_mol_s=(1.0 + target) * 1.0e-8,
                    target_current_density_a_m2=signed_target,
                    current_relative_error=0.02,
                    stable_samples=0,
                    required_stable_samples=config.stability_samples,
                    accepted=False,
                )
            )

        point = summarize_current_sweep(samples, config)[1]

        self.assertEqual(point.window_sample_count, config.stability_samples)
        self.assertAlmostEqual(point.window_start_s, first_stable_time)
        self.assertAlmostEqual(point.mean_measured_current_density_a_cm2, target)

    def test_csv_round_trip_and_figures(self) -> None:
        config = CurrentSweepConfig()
        samples = sweep_samples(config)
        points = summarize_current_sweep(samples, config)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            timeseries = output / "data/timeseries.csv"
            write_current_sweep_timeseries(timeseries, samples, config.membrane_area_m2)
            restored = read_current_sweep_timeseries(timeseries)
            restored_points = summarize_current_sweep(restored, config)
            write_current_sweep_points(output / "current_sweep.csv", restored_points)
            write_current_sweep_reports(
                output, restored, restored_points, config.membrane_area_m2
            )
            csv_text = (output / "current_sweep.csv").read_text(encoding="utf-8")
            self.assertIn("mean_crossover_flux_density_mol_m2_s", csv_text)
            for filename in ("current_sweep.png", "current_sweep_timeseries.png"):
                self.assertTrue((output / filename).read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
