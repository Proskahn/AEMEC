from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from aemec_opt.reporting import ReportSpec, TrialRecord, write_results


class ReportingTests(unittest.TestCase):
    def test_writes_completed_records_and_pareto_artifacts(self) -> None:
        records = (
            TrialRecord(0, 10.0, (1.0, 3.0), {"mode": "fast"}),
            TrialRecord(1, 20.0, (2.0, 2.0), {"mode": "fast"}),
            TrialRecord(2, 30.0, (3.0, 3.0), {"mode": "fast"}),
        )
        spec = ReportSpec(
            parameter_name="membrane_thickness_um",
            parameter_label="Membrane thickness [um]",
            objective_names=("cell_voltage_v", "crossover_mol_s"),
            objective_labels=("Cell voltage [V]", "Crossover [mol/s]"),
            title="Test Pareto front",
            metadata_columns=("mode",),
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_results(records, output, spec, ("minimize", "minimize"))
            self.assertTrue((output / "optimization_results.csv").is_file())
            self.assertTrue((output / "pareto_front.csv").is_file())
            self.assertTrue((output / "pareto_front.png").is_file())
            with (output / "pareto_front.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["trial"] for row in rows], ["0", "1"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
