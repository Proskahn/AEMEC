from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crossover_experiments.config import ExperimentConfig
from crossover_experiments.project import DEFAULT_SOURCE_CASE
from crossover_experiments.runner import ExperimentRunner, REQUIRED_MESH_REGIONS


def solver_log() -> str:
    lines = []
    for time_s, accepted in ((15.0, "false"), (20.0, "true")):
        lines.extend(
            (
                f"Time = {time_s}",
                "Controlled boundary current (A) at interconnect0: signed = -0.8, magnitude = 0.8, current density = -10000 A/m2, voltage = 1.9",
                "Hydrogen crossover objective: anode gas source rate = 4e-8 mol/s",
                (
                    "galvanostatic target: -10000 A/m2, measured current density: -10000 A/m2, "
                    "current relative error: 0, voltage: 1.9, stability samples: 5/5, "
                    f"accepted: {accepted}"
                ),
            )
        )
    lines.append("End")
    return "\n".join(lines)


class RunnerTests(unittest.TestCase):
    def test_full_artifact_flow_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "results"
            work = root / "work"
            calls: list[tuple[str, ...]] = []

            def fake_run(command, cwd, log_path, timeout_s):
                command = tuple(command)
                calls.append(command)
                if command == ("fake-mesh",):
                    mesh_paths = [cwd / "constant/polyMesh/points"] + [
                        cwd / "constant" / region / "polyMesh/points"
                        for region in REQUIRED_MESH_REGIONS
                    ]
                    for path in mesh_paths:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("points\n", encoding="utf-8")
                    output_text = "mesh complete"
                else:
                    output_text = solver_log()
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(output_text, encoding="utf-8")
                return output_text

            config = ExperimentConfig(thicknesses_um=(20.0,))
            runner = ExperimentRunner(
                source_case=DEFAULT_SOURCE_CASE,
                work_dir=work,
                output_dir=output,
                mesh_command=("fake-mesh",),
                solver_command=("fake-solver",),
                config=config,
                timeout_s=None,
            )
            with patch("crossover_experiments.runner.case_fingerprint", return_value="fingerprint"), patch(
                "crossover_experiments.runner.run_command", side_effect=fake_run
            ):
                self.assertEqual(runner.run(), 0)
            self.assertEqual(calls, [("fake-mesh",), ("fake-solver",)])
            for path in (
                output / "experiment.json",
                output / "cases/20um.json",
                output / "data/20um_timeseries.csv",
                output / "summary.csv",
                output / "timeseries.png",
                output / "thickness_summary.png",
            ):
                self.assertTrue(path.is_file(), path)

            calls.clear()
            resumed = ExperimentRunner(
                source_case=DEFAULT_SOURCE_CASE,
                work_dir=work,
                output_dir=output,
                mesh_command=("fake-mesh",),
                solver_command=("fake-solver",),
                config=config,
                timeout_s=None,
            )
            with patch("crossover_experiments.runner.case_fingerprint", return_value="fingerprint"), patch(
                "crossover_experiments.runner.run_command", side_effect=fake_run
            ):
                self.assertEqual(resumed.run(), 0)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()

