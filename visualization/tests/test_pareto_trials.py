from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from visualization.pareto_trials import (
    load_trials,
    plot_trials,
    select_first_trials,
)


class ParetoTrialPlotTests(unittest.TestCase):
    def test_load_select_and_plot_trials(self) -> None:
        csv_text = """trial,membrane_thickness_um,cell_voltage_v,crossover_rate_mol_s
0,10,1.8,6e-8
3,40,1.9,5e-8
1,20,1.85,5.5e-8
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "trials.csv"
            output_path = root / "pareto.png"
            csv_path.write_text(csv_text, encoding="utf-8")

            selected = select_first_trials(load_trials(csv_path), count=2)
            plot_trials(selected, output_path, count=2)

            self.assertEqual([trial.number for trial in selected], [0, 1])
            self.assertGreater(output_path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
