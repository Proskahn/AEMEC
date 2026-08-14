from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crossover_experiments.config import CurrentSweepConfig
from crossover_experiments.current_sweep_runner import CurrentSweepRunner
from crossover_experiments.project import DEFAULT_SOURCE_CASE
from crossover_experiments.runner import REQUIRED_MESH_REGIONS


def fake_solver_log(config: CurrentSweepConfig) -> str:
    lines: list[str] = []
    for target_index, (target, signed_target) in enumerate(
        zip(config.current_targets_a_cm2, config.signed_targets_a_m2)
    ):
        for sample_index in range(5):
            time_s = target_index * 20.0 + 19.6 + sample_index * 0.1
            current_a = signed_target * config.membrane_area_m2
            accepted = "true" if sample_index == 4 else "false"
            lines.extend(
                (
                    f"Time = {time_s}",
                    (
                        "Controlled boundary current (A) at interconnect0: "
                        f"signed = {current_a}, magnitude = {abs(current_a)}, "
                        f"current density = {signed_target} A/m2, "
                        f"voltage = {1.3 + 0.3 * target}"
                    ),
                    (
                        "Hydrogen crossover objective: anode gas source rate = "
                        f"{(1.0 + target) * 1.0e-8} mol/s"
                    ),
                    (
                        f"galvanostatic target: {signed_target} A/m2, "
                        f"measured current density: {signed_target} A/m2, "
                        "current relative error: 0, "
                        f"voltage: {1.3 + 0.3 * target}, "
                        f"stability samples: {sample_index + 1}/5, accepted: {accepted}"
                    ),
                )
            )
    lines.append("End")
    return "\n".join(lines)


class CurrentSweepRunnerTests(unittest.TestCase):
    def test_simulated_end_to_end_sweep(self) -> None:
        config = CurrentSweepConfig()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls: list[tuple[str, ...]] = []

            def fake_run(command, cwd, log_path, timeout_s):
                command = tuple(command)
                calls.append(command)
                if command == ("fake-mesh",):
                    paths = [cwd / "constant/polyMesh/points"] + [
                        cwd / "constant" / region / "polyMesh/points"
                        for region in REQUIRED_MESH_REGIONS
                    ]
                    for path in paths:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("points\n", encoding="utf-8")
                    output = "mesh complete"
                else:
                    output = fake_solver_log(config)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(output, encoding="utf-8")
                return output

            runner = CurrentSweepRunner(
                source_case=DEFAULT_SOURCE_CASE,
                work_dir=root / "work",
                output_dir=root / "results",
                mesh_command=("fake-mesh",),
                solver_command=("fake-solver",),
                config=config,
                timeout_s=None,
            )
            with patch(
                "crossover_experiments.current_sweep_runner.case_fingerprint",
                return_value="fingerprint",
            ), patch(
                "crossover_experiments.current_sweep_runner.run_command",
                side_effect=fake_run,
            ):
                self.assertEqual(runner.run(), 0)
            self.assertEqual(calls, [("fake-mesh",), ("fake-solver",)])
            for relative in (
                "experiment.json",
                "case.json",
                "data/timeseries.csv",
                "current_sweep.csv",
                "current_sweep.png",
                "current_sweep_timeseries.png",
            ):
                self.assertTrue((root / "results" / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()

