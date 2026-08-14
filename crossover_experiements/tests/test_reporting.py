from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crossover_experiments.parsing import ExperimentSample, summarize_samples
from crossover_experiments.reporting import write_reports


def samples(scale: float) -> list[ExperimentSample]:
    return [
        ExperimentSample(
            time_s=time_s,
            current_a=-0.8,
            current_density_a_m2=-10000.0,
            voltage_v=1.8 + 0.1 * scale,
            crossover_rate_mol_s=scale * (3.0 + time_s / 20.0) * 1.0e-8,
            target_current_density_a_m2=-10000.0,
            current_relative_error=0.0,
            stable_samples=5,
            required_stable_samples=5,
            accepted=True,
        )
        for time_s in (15.0, 17.5, 20.0)
    ]


class ReportingTests(unittest.TestCase):
    def test_writes_summary_and_both_png_figures(self) -> None:
        data = {20.0: samples(1.0), 80.0: samples(0.4)}
        summaries = [summarize_samples(key, value, 5.0) for key, value in data.items()]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_reports(output, data, summaries, 1.0)
            self.assertIn("thickness_um", (output / "summary.csv").read_text(encoding="utf-8"))
            for filename in ("timeseries.png", "thickness_summary.png"):
                self.assertTrue((output / filename).read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()

